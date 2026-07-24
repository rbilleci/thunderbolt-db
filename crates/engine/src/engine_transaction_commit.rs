//! Atomic explicit-transaction apply and GPU-residency publication helpers.

use super::*;
use crate::engine_transaction_catalog::TransactionCatalogEnvelopeSlices;
use crate::engine_transaction_reset::{
    table_access_dependency_identities, table_reset_empty_digest, table_schema_digest,
};

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

#[derive(Default)]
struct TransactionTableMutationBatch {
    old_rows: Vec<Vec<SqlValue>>,
    old_row_ids: Vec<u64>,
    new_rows: Vec<Vec<SqlValue>>,
    new_row_ids: Vec<u64>,
}

impl Engine {
    fn publish_transaction_cold_index_enrollment(&self, table: &RelationalTable) {
        self.purge_chunk_key_indexes_for_table(&table.name);
        self.purge_chunk_key_blooms_for_table(&table.name);
        // Neither chunk-authoritative nor store-authoritative cold residency retains a
        // per-index device directory between calls.  This OID-keyed entry is the exact enrollment
        // intent consumed if the same catalog/table identity later becomes hot again.
        let mut publications = self
            .read_state
            .residency
            .named_index_publications
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if table.indexes.is_empty() {
            publications.remove(&table.oid);
        } else {
            publications.insert(table.oid, table.indexes.clone());
        }
        drop(publications);
        self.read_state
            .residency
            .named_index_coverage_complete
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&table.name);
    }

    pub(crate) fn maintain_transaction_index_lifecycle_residency(
        &self,
        cat: &DdlCatalogState,
        tables: &BTreeSet<String>,
        publish_index: Index,
    ) -> Result<BTreeSet<String>, EngineError> {
        let mut maintained = BTreeSet::new();
        let resident_shards = self.read_residency_shards();
        let cold_chunks = self.read_streaming_cold_chunks();
        for table_name in tables {
            let table = cat.relational_catalog.get(table_name).ok_or_else(|| {
                EngineError::Durability(format!(
                    "transaction index lifecycle lost owner relation \"{table_name}\""
                ))
            })?;
            let without_resident_shards = resident_shards
                .get(table_name)
                .is_none_or(|shards| shards.is_empty());
            let exact_zero_row_generation = without_resident_shards
                && self.zero_row_resident_generation_boundary(table).is_some();
            let device_authoritative = self.table_device_authoritative(table_name);
            if device_authoritative && without_resident_shards && !exact_zero_row_generation {
                return Err(EngineError::ApplyFailed(format!(
                    "transaction index owner \"{table_name}\" lost its device-authoritative resident generation after WAL"
                )));
            }
            let cold_without_resident_shards = !device_authoritative
                && cold_chunks.contains_key(table_name)
                && without_resident_shards;
            if self.table_chunk_authoritative(table_name).is_some() || cold_without_resident_shards
            {
                self.publish_transaction_cold_index_enrollment(table);
                maintained.insert(table_name.clone());
                continue;
            }
            if exact_zero_row_generation {
                // A table created by this same transaction publishes a real zero-row device
                // generation before its index metadata. Index-only DDL is data-neutral, so advance
                // that immutable empty generation through this publication boundary before marking
                // coverage. There are no shard keys to build and no host rows to consult.
                self.read_state
                    .residency
                    .with_snapshots_mut(|snapshots| {
                        let entry = snapshots.get_mut(table_name).ok_or_else(|| {
                            EngineError::ApplyFailed(format!(
                                "transaction index owner \"{table_name}\" lost its zero-row resident generation"
                            ))
                        })?;
                        let descriptor = Arc::make_mut(&mut entry.descriptor);
                        if descriptor.row_count != 0 || entry.device_memory.is_none() {
                            return Err(EngineError::ApplyFailed(format!(
                                "transaction index owner \"{table_name}\" changed its zero-row resident generation"
                            )));
                        }
                        descriptor.valid_through_index =
                            descriptor.valid_through_index.max(publish_index);
                        Ok(())
                    })?;
                self.read_state
                    .residency
                    .purge_shard_pk_index_for_table_during_transaction(table_name);
                let mut publications = self
                    .read_state
                    .residency
                    .named_index_publications
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if table.indexes.is_empty() {
                    publications.remove(&table.oid);
                } else {
                    publications.insert(table.oid, table.indexes.clone());
                }
                drop(publications);
                let mut coverage = self
                    .read_state
                    .residency
                    .named_index_coverage_complete
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if table.indexes.is_empty() {
                    coverage.remove(table_name);
                } else {
                    coverage.insert(table_name.clone(), (table.oid, table.indexes.clone()));
                }
                maintained.insert(table_name.clone());
                continue;
            }
            let shards = resident_shards.get(table_name).cloned().ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "transaction index owner \"{table_name}\" lost its resident generation"
                ))
            })?;
            if table.indexes.is_empty() {
                self.read_state
                    .residency
                    .purge_shard_pk_index_for_table_during_transaction(table_name);
                self.read_state
                    .residency
                    .named_index_publications
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .remove(&table.oid);
            } else if shards.iter().all(|shard| shard.row_count == 0) {
                self.read_state
                    .residency
                    .purge_shard_pk_index_for_table_during_transaction(table_name);
                self.read_state
                    .residency
                    .named_index_coverage_complete
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .insert(table_name.clone(), (table.oid, table.indexes.clone()));
                self.read_state
                    .residency
                    .named_index_publications
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .insert(table.oid, table.indexes.clone());
            } else {
                self.publish_relational_resident_indexes_for_generation(
                    table,
                    &shards,
                    publish_index,
                    true,
                    false,
                    true,
                )
                .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
                let retained_key_ids = table
                    .indexes
                    .iter()
                    .enumerate()
                    .filter_map(|(ordinal, index)| {
                        crate::engine_residency::index_probe_key_id(table, index, ordinal)
                    })
                    .collect::<BTreeSet<_>>();
                self.read_state
                    .residency
                    .retain_shard_pk_indexes_for_table_during_transaction(
                        table_name,
                        &retained_key_ids,
                    );
            }
            maintained.insert(table_name.clone());
        }
        Ok(maintained)
    }

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
            AppliedRowMutation::TableReset { reset, .. } => reset.table.clone(),
            AppliedRowMutation::Insert { table, .. }
            | AppliedRowMutation::Delete { table, .. }
            | AppliedRowMutation::Update { table, .. } => table.clone(),
        }
    }

    /// Maintain every row operation from one atomic transaction against the current resident
    /// generation before publishing its shared commit sequence. Tables are independent, but every
    /// touched table must publish successfully: an incremental decline wedges the live commit path
    /// before acknowledgement, leaving typed WAL replay as the recovery source.
    pub(crate) fn try_maintain_transaction_residency(
        &self,
        cat: &mut DdlCatalogState,
        applied: &[AppliedRowMutation],
        publish_index: Index,
        transaction_created_tables: &BTreeSet<String>,
    ) -> Result<BTreeSet<String>, EngineError> {
        let table_resets = applied
            .iter()
            .filter_map(|mutation| match mutation {
                AppliedRowMutation::TableReset { reset, .. } => {
                    Some((reset.table.clone(), reset.clone()))
                }
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        let incremental = applied
            .iter()
            .filter(|mutation| !table_resets.contains_key(&Self::applied_mutation_table(mutation)))
            .cloned()
            .collect::<Vec<_>>();
        // Chunk-authoritative tables are transaction-wide COW: build every mutation against one
        // captured entry per table and publish the complete replacement map once. They must never
        // enter the legacy per-mutation publisher loop below.
        let cold_maintained =
            self.apply_transaction_cold_batch(cat, &incremental, publish_index)?;
        let mut status: BTreeMap<String, bool> = cold_maintained
            .iter()
            .map(|table| (table.clone(), true))
            .collect();
        // A relation created by this same atomic transaction has no globally-published base shard.
        // Its coalesced final mutation chain can only contain INSERTs (private insert→update folds
        // to one final INSERT; insert→delete vanishes). Build and install that first device
        // generation directly from the resolved record, which gives live apply and recovery the
        // identical GPU-native path.
        let current_shards = self.read_residency_shards();
        let mut new_tables = BTreeMap::<String, (Vec<Vec<SqlValue>>, Vec<u64>)>::new();
        let fresh_tables = transaction_created_tables
            .iter()
            .cloned()
            .chain(table_resets.keys().cloned())
            .collect::<BTreeSet<_>>();
        for table_name in transaction_created_tables {
            if current_shards
                .get(table_name)
                .is_none_or(|shards| shards.is_empty())
            {
                // CREATE TABLE with no surviving private INSERT still owns a typed, allocation-
                // backed zero-row GPU generation. This lets catalog/index publication close over
                // an exact device authority instead of treating missing shards as empty by fiat.
                new_tables.entry(table_name.clone()).or_default();
            }
        }
        for mutation in applied {
            let table_name = Self::applied_mutation_table(mutation);
            if !fresh_tables.contains(&table_name) {
                continue;
            }
            if !table_resets.contains_key(&table_name)
                && current_shards
                    .get(&table_name)
                    .is_some_and(|shards| !shards.is_empty())
            {
                continue;
            }
            match mutation {
                AppliedRowMutation::TableReset { .. } => {
                    new_tables.entry(table_name).or_default();
                }
                AppliedRowMutation::Insert {
                    rows, row_ids, ..
                } => {
                    let entry = new_tables.entry(table_name).or_default();
                    entry.0.extend(rows.iter().cloned());
                    entry.1.extend(row_ids.iter().copied());
                }
                AppliedRowMutation::Update { .. } | AppliedRowMutation::Delete { .. } => {
                    return Err(EngineError::Durability(format!(
                        "transaction WAL mutates relation \"{table_name}\" without a published or create-owned base generation"
                    )))
                }
            }
        }
        let mut freshly_installed = BTreeSet::new();
        for (table_name, (rows, row_ids)) in new_tables {
            let table = cat.relational_catalog.get(&table_name).cloned().ok_or_else(|| {
                EngineError::Durability(format!(
                    "transaction WAL lost create-owned relation \"{table_name}\" before residency publication"
                ))
            })?;
            if table_resets.contains_key(&table_name) {
                self.prepare_committed_table_reset(cat, &table_name)?;
            }
            self.install_resolved_transaction_table_residency(
                cat,
                &table,
                &rows,
                &row_ids,
                publish_index,
                table_resets.contains_key(&table_name),
            )?;
            status.insert(table_name.clone(), true);
            freshly_installed.insert(table_name);
        }
        // The WAL record has already coalesced each logical entity to its final mutation. Build one
        // table batch from that record, stamp all old versions first, then append all final images
        // once. This is both the atomic publication shape and the exact geometry reserved before
        // WAL: statement-by-statement rollover is not a second transaction write path.
        let mut batches = BTreeMap::<String, TransactionTableMutationBatch>::new();
        for mutation in applied {
            let table = Self::applied_mutation_table(mutation);
            if cold_maintained.contains(&table) || freshly_installed.contains(&table) {
                continue;
            }
            let batch = batches.entry(table.clone()).or_default();
            match mutation {
                AppliedRowMutation::TableReset { .. } => {
                    unreachable!("fresh table resets are installed above")
                }
                AppliedRowMutation::Insert { rows, row_ids, .. } => {
                    batch.new_rows.extend(rows.iter().cloned());
                    batch.new_row_ids.extend(row_ids.iter().copied());
                }
                AppliedRowMutation::Delete {
                    rows, write_set, ..
                } => {
                    let prefix = relational_key_prefix(&table);
                    let row_ids = write_set
                        .rows
                        .iter()
                        .map(|key| {
                            crate::engine_residency::parse_relational_row_id(&key.row_key, &prefix)
                        })
                        .collect::<Option<Vec<_>>>();
                    let Some(row_ids) = row_ids else {
                        return Err(EngineError::Durability(format!(
                            "transaction WAL delete lost an entity identity for relation \"{table}\""
                        )));
                    };
                    batch.old_rows.extend(rows.iter().cloned());
                    batch.old_row_ids.extend(row_ids);
                }
                AppliedRowMutation::Update {
                    old_rows,
                    new_rows,
                    row_ids,
                    ..
                } => {
                    let Some(row_ids) = row_ids else {
                        return Err(EngineError::Durability(format!(
                            "transaction WAL update lost entity identities for relation \"{table}\""
                        )));
                    };
                    batch.old_rows.extend(old_rows.iter().cloned());
                    batch.old_row_ids.extend(row_ids.iter().copied());
                    batch.new_rows.extend(new_rows.iter().cloned());
                    batch.new_row_ids.extend(row_ids.iter().copied());
                }
            }
        }
        for (table, batch) in batches {
            let arity_ok = batch.old_rows.len() == batch.old_row_ids.len()
                && batch.new_rows.len() == batch.new_row_ids.len();
            let tombstoned = arity_ok
                && (batch.old_rows.is_empty()
                    || self.try_tombstone_transaction_rows_by_identity(
                        cat,
                        &table,
                        &batch.old_rows,
                        &batch.old_row_ids,
                        publish_index,
                    ));
            let appended = batch.new_rows.is_empty()
                || self.try_append_resident_int4_open_shard(
                    &table,
                    &batch.new_rows,
                    crate::engine_residency::AppendCreatedBy::InsertUniform(publish_index),
                    Some(&batch.new_row_ids),
                );
            status.insert(table, tombstoned && appended);
        }

        let mut maintained = BTreeSet::new();
        for (table_name, table_handled) in status {
            if table_handled {
                maintained.insert(table_name.clone());
                if self.table_chunk_authoritative(&table_name).is_none()
                    && !self.table_device_authoritative(&table_name)
                {
                    let snapshot = self.catalog_snapshot();
                    if self.table_device_authority_eligible(&snapshot, &table_name) {
                        self.set_table_device_authoritative(&table_name, true);
                    }
                }
            } else {
                self.wedge_commit_path();
                return Err(EngineError::ApplyFailed(format!(
                    "durable transaction DML for relation \"{table_name}\" could not publish its device generation at {publish_index}"
                )));
            }
        }
        self.read_state.publish_table_rewrite_fences(
            table_resets.values().map(|reset| reset.table_oid),
            publish_index,
        );
        Ok(maintained)
    }

    fn install_resolved_transaction_table_residency(
        &self,
        cat: &mut DdlCatalogState,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
        row_ids: &[u64],
        publish_index: Index,
        reset_root: bool,
    ) -> Result<(), EngineError> {
        if rows.is_empty() {
            self.populate_relational_residency_snapshot_inner_with_boundary(
                cat,
                &table.name,
                self.planner.default_gpu_id(),
                Some(publish_index),
            )
            .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
            let maintained = self.maintain_transaction_index_lifecycle_residency(
                cat,
                &BTreeSet::from([table.name.clone()]),
                publish_index,
            )?;
            if !maintained.contains(&table.name) {
                return Err(EngineError::ApplyFailed(format!(
                    "resolved transaction did not enroll the empty relation \"{}\" index generation",
                    table.name
                )));
            }
            return Ok(());
        }
        let mut private_map = BTreeMap::from([(table.name.clone(), Vec::new())]);
        let mut reservation = crate::engine_transaction_delta::TransactionGpuReservation::new(self);
        self.append_transaction_delta_shard(
            table,
            rows,
            row_ids,
            &mut private_map,
            &mut reservation,
        )
        .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
        let mut shards = private_map.remove(&table.name).ok_or_else(|| {
            EngineError::ApplyFailed(format!(
                "resolved transaction lost new relation \"{}\" device generation",
                table.name
            ))
        })?;
        // A reset discards the complete prior physical/version history. Its replacement shards
        // therefore begin at the reset publication, not at the private builder's additive-history
        // sentinel. Older writers must observe the rewrite fence or decline on this floor.
        if reset_root {
            for shard in &mut shards {
                shard.history_floor_index = publish_index;
            }
        }
        let device_memory = shards
            .iter()
            .filter_map(|shard| {
                shard
                    .device_memory
                    .as_ref()
                    .map(|memory| (shard.shard_id, Arc::clone(memory)))
            })
            .collect::<BTreeMap<_, _>>();
        cat.relational_resident_cache.install_shards(
            table.name.clone(),
            shards,
            device_memory,
            &self.read_state.residency,
        );
        // The installed allocation is now counted by the global resident maps. Remove the
        // temporary pre-allocation charge before the mandatory named-index budget transaction.
        drop(reservation);
        // A transaction-created relation's declared PK/UNIQUE indexes are part of its first
        // canonical device generation, not optional cache warm-up. Commit reserved their exact
        // bytes before WAL; live apply and recovery must therefore publish complete coverage even
        // though this new OID had no prior explicit enrollment marker.
        if !table.indexes.is_empty() {
            let current = self.read_residency_shards();
            let shards = current.get(&table.name).ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "resolved transaction lost new relation \"{}\" before index publication",
                    table.name
                ))
            })?;
            self.publish_relational_resident_indexes_for_generation(
                table,
                shards,
                publish_index,
                true,
                false,
                true,
            )
            .map_err(|error| EngineError::ApplyFailed(error.to_string()))?;
        }
        Ok(())
    }

    fn prepare_committed_table_reset(
        &self,
        cat: &mut DdlCatalogState,
        table_name: &str,
    ) -> Result<(), EngineError> {
        self.set_table_device_authoritative(table_name, false);
        let residency = &self.read_state.residency;
        {
            let mut authority = (**residency.chunk_authoritative_tables.load()).clone();
            authority.remove(table_name);
            residency
                .chunk_authoritative_tables
                .store(Arc::new(authority));
        }
        {
            let _cold = residency
                .streaming_cold_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut chunks = (**residency.streaming_cold_chunks.load()).clone();
            chunks.remove(table_name);
            residency.streaming_cold_chunks.store(Arc::new(chunks));
        }
        self.purge_chunk_key_indexes_for_table(table_name);
        self.purge_chunk_key_blooms_for_table(table_name);
        cat.relational_resident_cache.remove_table(
            table_name,
            residency,
            &self.read_state.route_telemetry,
        );
        self.read_state.mvcc.publish_empty_table(table_name);
        Ok(())
    }

    /// Transaction commit already carries exact stable entity identities, so it need not fall back
    /// to the legacy all-non-NULL int4 row predicate. Probe a device key, verify identity plus the
    /// complete nullable row image, and stamp exactly that visible physical version.
    pub(crate) fn try_tombstone_transaction_rows_by_identity(
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
        self.try_tombstone_rows_by_identity(table, rows, row_ids, commit_seq)
    }

    /// Stamp exact physical versions using the stable entity identities carried by every resolved
    /// device DML delta. This is shared by explicit-transaction publication and autocommit
    /// UPDATE/DELETE maintenance: nullable or otherwise non-legacy row shapes must never fall back
    /// to an all-non-NULL int4 image predicate after the durable cut.
    pub(crate) fn try_tombstone_rows_by_identity(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
        row_ids: &[u64],
        commit_seq: Index,
    ) -> bool {
        let Some(locations) =
            self.locate_transaction_rows_by_identity(table, rows, row_ids, commit_seq)
        else {
            return false;
        };
        for (shard_id, slots) in locations {
            if !self.tombstone_resident_shard_slots(&table.name, shard_id, &slots, commit_seq) {
                return false;
            }
        }
        self.add_tombstone_churn(&table.name, rows.len() as u64);
        true
    }

    /// Resolve the exact physical versions for a coalesced transaction without mutating them.
    /// COMMIT preflight and canonical apply share this resolver, so first-delete sidecar sizing is
    /// based on the same shard identities that publication will stamp after durability.
    pub(crate) fn locate_transaction_rows_by_identity(
        &self,
        table: &RelationalTable,
        rows: &[Vec<SqlValue>],
        row_ids: &[u64],
        visibility: Index,
    ) -> Option<BTreeMap<u32, Vec<u32>>> {
        if rows.len() != row_ids.len() {
            return None;
        }
        let mut locations = BTreeMap::<u32, Vec<u32>>::new();
        for (row, expected_row_id) in rows.iter().zip(row_ids) {
            let filters = row
                .iter()
                .enumerate()
                .map(|(idx, value)| (idx, SelectFilterOp::Eq, value.clone()))
                .collect::<Vec<_>>();
            let indexed =
                self.dml_device_probe_key(table, &filters)
                    .and_then(|(key_id, needle)| {
                        self.locate_resident_pk_via_shard_index_detailed(table, key_id, needle)
                    });
            let hits = match indexed {
                Some(hits) if !hits.is_empty() => hits,
                _ => {
                    // UPDATE version churn deliberately makes a cached unique index decline once
                    // the old and new physical slots share a key. Re-resolve the complete old row
                    // structurally on-device, then identity/materialization-recheck below; never
                    // route that ordinary churn through host reconstruction.
                    let key_cols = row
                        .iter()
                        .enumerate()
                        .map(|(idx, value)| (idx, value.clone()))
                        .collect::<Vec<_>>();
                    let predicate = crate::engine_dml_prepare::device_structural_tuple_predicate(
                        table, &key_cols,
                    )?;
                    self.locate_resident_delete_slots_detailed(table, &predicate)?
                }
            };
            let mut matched = Vec::new();
            for hit in hits {
                if self.hit_entity_id(&hit) != Some(*expected_row_id) {
                    continue;
                }
                match self.materialize_resident_row_via_hit(table, &hit, visibility) {
                    Some(Some(observed)) if observed.as_slice() == row.as_slice() => {
                        matched.push((hit.shard_id, hit.slot));
                    }
                    Some(Some(_)) | Some(None) => {}
                    None => return None,
                }
            }
            if matched.len() != 1 {
                return None;
            }
            let (shard_id, slot) = matched[0];
            locations.entry(shard_id).or_default().push(slot);
        }
        Some(locations)
    }

    pub(crate) fn apply_binary_transaction_record(
        &self,
        entry: &LogEntry,
        cat: &mut DdlCatalogState,
        record: BinaryTransactionRecord,
    ) -> Result<Vec<AppliedRowMutation>, EngineError> {
        let BinaryTransactionRecord {
            catalog_epoch,
            allocator_high_water,
            catalog_commands,
            created_table_identities,
            created_table_index_identities,
            catalog_output,
            view_operations,
            view_lifecycle_operations,
            index_lifecycle_operations,
            sequence_lifecycle_operations,
            sequence_reset_operations,
            sequence_advances_by_oid,
            operation_order,
            statement_digests,
            sequence_input_oids,
            sequence_value_references,
            sequence_advances,
            table_identities,
            mutations,
            table_resets,
        } = record;

        self.validate_sequence_value_references(&sequence_value_references)?;

        if catalog_commands.len() > 1 && (catalog_output.is_none() || operation_order.is_empty()) {
            return Err(EngineError::Durability(
                "ordered transaction WAL lost its statement or catalog identity closure"
                    .to_string(),
            ));
        }
        if (!view_operations.is_empty()
            || !view_lifecycle_operations.is_empty()
            || !index_lifecycle_operations.is_empty()
            || !sequence_lifecycle_operations.is_empty()
            || !sequence_reset_operations.is_empty()
            || !sequence_advances_by_oid.is_empty())
            && (catalog_output.is_none() || operation_order.is_empty())
        {
            return Err(EngineError::Durability(
                "transactional catalog-lifecycle WAL lost its ordered catalog envelope".to_string(),
            ));
        }

        let ordered_envelope = !operation_order.is_empty();
        if ordered_envelope != catalog_output.is_some() {
            return Err(EngineError::Durability(
                "ordered transaction WAL lost its catalog output closure".to_string(),
            ));
        }
        if ordered_envelope {
            if statement_digests.len() != operation_order.len()
                || statement_digests.contains(&[0; 32])
            {
                return Err(EngineError::Durability(
                    "ordered transaction statement digests do not cover its operation order"
                        .to_string(),
                ));
            }
        } else if !statement_digests.is_empty() || !sequence_input_oids.is_empty() {
            return Err(EngineError::Durability(
                "legacy transaction WAL carries ordered-only statement identity fields".to_string(),
            ));
        }
        let created_ordinals = catalog_commands
            .iter()
            .filter_map(|operation| match &operation.command {
                Command::CreateTable(create) => Some((create.table.clone(), operation.ordinal)),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        let created_sequence_names = catalog_commands
            .iter()
            .flat_map(|operation| match &operation.command {
                Command::CreateTable(create) => create
                    .columns
                    .iter()
                    .filter_map(|column| match &column.default {
                        Some(ColumnDefault::SequenceNextVal {
                            sequence,
                            create_if_missing: true,
                        }) => Some(sequence.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect::<Vec<_>>();
        if created_sequence_names.iter().collect::<BTreeSet<_>>().len()
            != created_sequence_names.len()
        {
            return Err(EngineError::Durability(
                "ordered transaction catalog repeats an implicit sequence identity".to_string(),
            ));
        }
        if let Some(output) = &catalog_output {
            let generated =
                generated_sequence_output_from_inputs(&catalog_commands, &sequence_input_oids)
                    .ok_or_else(|| {
                        EngineError::Durability(
                            "ordered generated sequence has no matching CREATE statement input"
                                .to_string(),
                        )
                    })?;
            if output.created_sequence_oids != generated {
                return Err(EngineError::Durability(
                    "ordered generated sequence output contradicts its CREATE statement sequence input"
                        .to_string(),
                ));
            }
        }
        let sequence_resets_by_ordinal = sequence_reset_operations
            .iter()
            .map(|identity| (identity.ordinal, identity))
            .collect::<BTreeMap<_, _>>();
        if sequence_resets_by_ordinal.len() != sequence_reset_operations.len() {
            return Err(EngineError::Durability(
                "ordered transaction repeats a sequence-reset statement ordinal".to_string(),
            ));
        }
        let mut ordered_row_names = BTreeSet::new();
        let mut ordered_reset_names = BTreeSet::new();
        if ordered_envelope {
            let mut next_catalog_index = 0usize;
            for (ordinal, operation) in operation_order.iter().enumerate() {
                let statement_digest = statement_digests.get(ordinal).ok_or_else(|| {
                    EngineError::Durability(
                        "ordered transaction operation has no statement digest".to_string(),
                    )
                })?;
                match operation {
                    BinaryTransactionOperationIdentity::Catalog { command_index } => {
                        if usize::try_from(*command_index).ok() != Some(next_catalog_index)
                            || catalog_commands
                                .get(next_catalog_index)
                                .is_none_or(|command| {
                                    usize::try_from(command.ordinal).ok() != Some(ordinal)
                                })
                        {
                            return Err(EngineError::Durability(
                                "ordered transaction catalog identity does not match its statement position"
                                    .to_string(),
                            ));
                        }
                        if transaction_statement_digest(
                            &catalog_commands[next_catalog_index].command,
                        )
                        .map_err(|error| EngineError::Durability(error.to_string()))?
                            != *statement_digest
                        {
                            return Err(EngineError::Durability(
                                "ordered transaction catalog statement digest mismatch".to_string(),
                            ));
                        }
                        next_catalog_index += 1;
                    }
                    BinaryTransactionOperationIdentity::Insert { table }
                    | BinaryTransactionOperationIdentity::Update { table }
                    | BinaryTransactionOperationIdentity::Delete { table } => {
                        if created_ordinals.get(table).is_some_and(|created| {
                            usize::try_from(*created)
                                .ok()
                                .is_none_or(|created| created >= ordinal)
                        }) {
                            return Err(EngineError::Durability(format!(
                                "ordered transaction row operation precedes relation \"{table}\""
                            )));
                        }
                        ordered_row_names.insert(table.clone());
                    }
                    BinaryTransactionOperationIdentity::TableReset { table } => {
                        if created_ordinals.get(table).is_some_and(|created| {
                            usize::try_from(*created)
                                .ok()
                                .is_none_or(|created| created >= ordinal)
                        }) {
                            return Err(EngineError::Durability(format!(
                                "ordered transaction reset precedes relation \"{table}\""
                            )));
                        }
                        let command = Command::TruncateTable(TruncateTable {
                            name: table.clone(),
                            restart_identity: u32::try_from(ordinal)
                                .ok()
                                .and_then(|ordinal| sequence_resets_by_ordinal.get(&ordinal))
                                .is_some_and(|identity| identity.table == *table),
                        });
                        if transaction_statement_digest(&command)
                            .map_err(|error| EngineError::Durability(error.to_string()))?
                            != *statement_digest
                        {
                            return Err(EngineError::Durability(
                                "ordered transaction table-reset statement digest mismatch"
                                    .to_string(),
                            ));
                        }
                        ordered_reset_names.insert(table.clone());
                    }
                }
            }
            if next_catalog_index != catalog_commands.len() {
                return Err(EngineError::Durability(
                    "ordered transaction statement envelope omits a catalog operation".to_string(),
                ));
            }
        }

        // Decode and bind the complete record before touching any globally published state. This
        // makes malformed replay/live records all-or-nothing and leaves only infallible assignment
        // plus prevalidated COW publication after the staging pass.
        let mut next_catalog = cat.clone();
        let created_names = created_ordinals
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if (ordered_envelope || !created_table_identities.is_empty())
            && created_names
                != created_table_identities
                    .keys()
                    .map(String::as_str)
                    .collect()
        {
            return Err(EngineError::Durability(
                "ordered transaction WAL created-table identities are incomplete".to_string(),
            ));
        }
        match catalog_epoch {
            BinaryTransactionCatalogEpoch::IndexIdentityV1
                if created_names
                    != created_table_index_identities
                        .keys()
                        .map(String::as_str)
                        .collect() =>
            {
                return Err(EngineError::Durability(
                    "current ordered transaction WAL created-table index identities are incomplete"
                        .to_string(),
                ));
            }
            BinaryTransactionCatalogEpoch::Legacy if !created_table_index_identities.is_empty() => {
                return Err(EngineError::Durability(
                    "legacy transaction WAL carries current created-table index identities"
                        .to_string(),
                ));
            }
            _ => {}
        }
        self.apply_transaction_catalog_envelope(
            &mut next_catalog,
            entry.index.saturating_sub(1),
            catalog_epoch,
            entry.index,
            &catalog_commands,
            TransactionCatalogEnvelopeSlices {
                view_operations: &view_operations,
                view_lifecycle_operations: &view_lifecycle_operations,
                index_lifecycle_operations: &index_lifecycle_operations,
                sequence_lifecycle_operations: &sequence_lifecycle_operations,
                sequence_reset_operations: &sequence_reset_operations,
                operation_order: &operation_order,
                catalog_output: catalog_output.as_ref(),
                sequence_input_oids: &sequence_input_oids,
                sequence_value_references: &sequence_value_references,
            },
        )?;
        let next_catalog_snapshot =
            Self::catalog_snapshot_from_working(&next_catalog, entry.index.saturating_sub(1));
        for reference in sequence_value_references
            .iter()
            .filter(|reference| reference.default_expression)
        {
            let table = next_catalog
                .relational_catalog
                .values()
                .find(|table| table.oid == reference.table_oid)
                .ok_or_else(|| {
                    EngineError::Durability(format!(
                        "sequence default reference {} targets unknown table identity {}",
                        reference.transition_txn_id, reference.table_oid
                    ))
                })?;
            let column = table
                .columns
                .iter()
                .find(|column| column.id == reference.column_id)
                .ok_or_else(|| {
                    EngineError::Durability(format!(
                        "sequence default reference {} targets unknown column identity {}",
                        reference.transition_txn_id, reference.column_id
                    ))
                })?;
            let Some(ColumnDefault::SequenceNextVal { sequence, .. }) = &column.default else {
                return Err(EngineError::Durability(format!(
                    "sequence default reference {} does not target a sequence-backed column",
                    reference.transition_txn_id
                )));
            };
            if next_catalog
                .relational_sequences
                .get(sequence)
                .map(|sequence| sequence.oid)
                != Some(reference.sequence_oid)
                || i32::try_from(reference.returned_value).is_err()
            {
                return Err(EngineError::Durability(format!(
                    "sequence default reference {} changed its stable dependency or materialized int4 value",
                    reference.transition_txn_id
                )));
            }
            let statement_table = operation_order
                .get(reference.statement_ordinal as usize)
                .and_then(|operation| match operation {
                    BinaryTransactionOperationIdentity::Insert { table } => Some(table.as_str()),
                    _ => None,
                });
            if (!operation_order.is_empty() && statement_table != Some(table.name.as_str()))
                || (operation_order.is_empty()
                    && !reference.final_value_overwritten
                    && !mutations.iter().any(|mutation| {
                        matches!(
                            mutation,
                            BinaryTransactionMutation::Insert {
                                table: mutation_table,
                                ..
                            } if mutation_table == &table.name
                        )
                    }))
            {
                return Err(EngineError::Durability(format!(
                    "sequence default reference {} is not bound to its INSERT target",
                    reference.transition_txn_id
                )));
            }
            let column_index = table
                .columns
                .iter()
                .position(|candidate| candidate.id == column.id)
                .expect("stable column was resolved above");
            let expected = SqlValue::Int4(
                i32::try_from(reference.returned_value).expect("int4 range was validated above"),
            );
            let mut matching_rows = mutations.iter().filter_map(|mutation| match mutation {
                BinaryTransactionMutation::Insert {
                    table: mutation_table,
                    row_id,
                    row_encoded,
                } if mutation_table == &table.name && *row_id == reference.row_id => {
                    Some(row_encoded)
                }
                BinaryTransactionMutation::Update {
                    table: mutation_table,
                    row_id,
                    new_row_encoded,
                    ..
                } if mutation_table == &table.name && *row_id == reference.row_id => {
                    Some(new_row_encoded)
                }
                BinaryTransactionMutation::Insert { .. }
                | BinaryTransactionMutation::Update { .. }
                | BinaryTransactionMutation::Delete { .. } => None,
            });
            let value_retained = match (matching_rows.next(), matching_rows.next()) {
                (Some(encoded), None) => {
                    let row = decode_relational_row(encoded, &table.columns).map_err(|error| {
                        EngineError::Durability(format!(
                            "sequence default reference {} final row is invalid: {error}",
                            reference.transition_txn_id
                        ))
                    })?;
                    row.get(column_index) == Some(&expected)
                }
                (None, None) => false,
                (Some(_), Some(_)) | (None, Some(_)) => {
                    return Err(EngineError::Durability(format!(
                        "sequence default reference {} has ambiguous final entity identity",
                        reference.transition_txn_id
                    )));
                }
            };
            if reference.final_value_overwritten == value_retained {
                return Err(EngineError::Durability(format!(
                    "sequence default reference {} does not match its final row disposition",
                    reference.transition_txn_id
                )));
            }
        }
        for (table_name, identity) in &created_table_identities {
            let table = next_catalog
                .relational_catalog
                .get(table_name)
                .ok_or_else(|| {
                    EngineError::Durability(format!(
                        "ordered transaction did not create relation \"{table_name}\""
                    ))
                })?;
            let schema_digest = table_schema_digest(table)
                .map_err(|error| EngineError::Durability(error.to_string()))?;
            if table.oid != identity.table_oid || schema_digest != identity.schema_digest {
                return Err(EngineError::Durability(format!(
                    "ordered transaction created relation \"{table_name}\" with a different stable identity"
                )));
            }
            if catalog_epoch == BinaryTransactionCatalogEpoch::IndexIdentityV1 {
                let observed = Self::transaction_created_table_implicit_index_identities(
                    &next_catalog_snapshot,
                    table_name,
                )
                .map_err(|error| EngineError::Durability(error.to_string()))?;
                if created_table_index_identities.get(table_name) != Some(&observed) {
                    return Err(EngineError::Durability(format!(
                        "ordered transaction created relation \"{table_name}\" with different implicit index identities"
                    )));
                }
            }
        }
        if let Some(output) = &catalog_output {
            if next_catalog.relational_next_oid != output.relational_next_oid
                || next_catalog.relational_next_column_id != output.relational_next_column_id
                || output.created_sequence_oids.keys().collect::<BTreeSet<_>>()
                    != created_sequence_names.iter().collect::<BTreeSet<_>>()
            {
                return Err(EngineError::Durability(
                    "ordered transaction catalog allocator or generated-sequence closure changed"
                        .to_string(),
                ));
            }
            for (sequence_name, expected_oid) in &output.created_sequence_oids {
                if !next_catalog
                    .relational_sequences
                    .values()
                    .any(|sequence| sequence.oid == *expected_oid)
                {
                    return Err(EngineError::Durability(format!(
                        "ordered transaction generated sequence \"{sequence_name}\" lost its stable identity"
                    )));
                }
            }
        }
        let mut validated_sequence_inputs = 0usize;
        let mut insert_sequence_names = BTreeSet::new();
        for (ordinal, operation) in operation_order.iter().enumerate() {
            let Some(table_name) = operation.table() else {
                if !matches!(
                    operation,
                    BinaryTransactionOperationIdentity::Catalog { .. }
                ) {
                    continue;
                }
                let ordinal = u32::try_from(ordinal).map_err(|_| {
                    EngineError::Durability(
                        "ordered transaction operation ordinal exceeds u32".to_string(),
                    )
                })?;
                let inputs = sequence_input_oids
                    .iter()
                    .filter(|((input_ordinal, _), _)| *input_ordinal == ordinal)
                    .collect::<Vec<_>>();
                validated_sequence_inputs += inputs.len();
                let BinaryTransactionOperationIdentity::Catalog { command_index } = operation
                else {
                    unreachable!("catalog operation checked above")
                };
                let command = &catalog_commands[usize::try_from(*command_index).map_err(|_| {
                    EngineError::Durability(
                        "ordered catalog command index exceeds usize".to_string(),
                    )
                })?]
                .command;
                let expected_names = match command {
                    Command::CreateTable(create) => create
                        .columns
                        .iter()
                        .filter_map(|column| match &column.default {
                            Some(ColumnDefault::SequenceNextVal { sequence, .. }) => {
                                Some(sequence.as_str())
                            }
                            _ => None,
                        })
                        .collect::<BTreeSet<_>>(),
                    Command::CreateView(_) | Command::RenameView(_) | Command::DropView(_) => {
                        BTreeSet::new()
                    }
                    Command::CreateIndex(_) | Command::RenameIndex(_) | Command::DropIndex(_) => {
                        BTreeSet::new()
                    }
                    Command::CreateSequence(_)
                    | Command::SequenceRestart(_)
                    | Command::RenameSequence(_)
                    | Command::DropSequence(_) => BTreeSet::new(),
                    _ => {
                        return Err(EngineError::Durability(
                            "ordered sequence closure names an unsupported catalog command"
                                .to_string(),
                        ))
                    }
                };
                let input_names = inputs
                    .iter()
                    .map(|((_, sequence), _)| sequence.as_str())
                    .collect::<BTreeSet<_>>();
                if input_names != expected_names {
                    return Err(EngineError::Durability(
                        "ordered catalog sequence identities do not cover its exact defaults"
                            .to_string(),
                    ));
                }
                for ((_, sequence_name), expected_oid) in inputs {
                    if !next_catalog
                        .relational_sequences
                        .values()
                        .any(|sequence| sequence.oid == *expected_oid)
                    {
                        return Err(EngineError::Durability(format!(
                            "ordered catalog sequence input \"{sequence_name}\" changed stable identity"
                        )));
                    }
                }
                continue;
            };
            let table = next_catalog
                .relational_catalog
                .get(table_name)
                .ok_or_else(|| {
                    EngineError::Durability(format!(
                        "ordered transaction operation targets unknown relation \"{table_name}\""
                    ))
                })?;
            let ordinal = u32::try_from(ordinal).map_err(|_| {
                EngineError::Durability(
                    "ordered transaction operation ordinal exceeds u32".to_string(),
                )
            })?;
            let inputs = sequence_input_oids
                .iter()
                .filter(|((input_ordinal, _), _)| *input_ordinal == ordinal)
                .collect::<Vec<_>>();
            validated_sequence_inputs += inputs.len();
            if matches!(operation, BinaryTransactionOperationIdentity::Insert { .. }) {
                let allowed_sequence_oids = table
                    .columns
                    .iter()
                    .filter_map(|column| match &column.default {
                        Some(ColumnDefault::SequenceNextVal { sequence, .. }) => next_catalog
                            .relational_sequences
                            .get(sequence)
                            .map(|sequence| sequence.oid),
                        _ => None,
                    })
                    .collect::<BTreeSet<_>>();
                for ((_, sequence_name), expected_oid) in inputs {
                    if !allowed_sequence_oids.contains(expected_oid)
                        || !next_catalog
                            .relational_sequences
                            .values()
                            .any(|sequence| sequence.oid == *expected_oid)
                    {
                        return Err(EngineError::Durability(format!(
                            "ordered INSERT sequence input \"{sequence_name}\" is not an exact default dependency of relation \"{table_name}\""
                        )));
                    }
                    insert_sequence_names.insert(sequence_name.as_str());
                }
            } else if !inputs.is_empty() {
                return Err(EngineError::Durability(
                    "non-INSERT ordered operation carries a sequence input identity".to_string(),
                ));
            }
        }
        if ordered_envelope {
            let insert_sequence_oids = sequence_input_oids
                .iter()
                .filter_map(|((ordinal, _), oid)| {
                    usize::try_from(*ordinal)
                        .ok()
                        .and_then(|ordinal| operation_order.get(ordinal))
                        .and_then(|operation| {
                            matches!(operation, BinaryTransactionOperationIdentity::Insert { .. })
                                .then_some(*oid)
                        })
                })
                .collect::<BTreeSet<_>>();
            if validated_sequence_inputs != sequence_input_oids.len()
                || if !sequence_advances_by_oid.is_empty()
                    || !sequence_lifecycle_operations.is_empty()
                    || !sequence_reset_operations.is_empty()
                {
                    !sequence_advances.is_empty()
                        || insert_sequence_oids
                            != sequence_advances_by_oid
                                .keys()
                                .copied()
                                .collect::<BTreeSet<_>>()
                } else {
                    insert_sequence_names
                        != sequence_advances
                            .keys()
                            .map(String::as_str)
                            .collect::<BTreeSet<_>>()
                }
            {
                return Err(EngineError::Durability(
                    "ordered INSERT sequence identities do not close over sequence advances"
                        .to_string(),
                ));
            }
        }
        if !ordered_envelope {
            for sequence_name in sequence_advances.keys() {
                if !next_catalog
                    .relational_sequences
                    .contains_key(sequence_name)
                {
                    return Err(EngineError::Durability(format!(
                        "transaction WAL record advances unknown sequence \"{sequence_name}\""
                    )));
                }
            }
        }
        for sequence_oid in sequence_advances_by_oid.keys() {
            let exists_after = next_catalog
                .relational_sequences
                .values()
                .any(|sequence| sequence.oid == *sequence_oid);
            let dropped_by_lifecycle = sequence_lifecycle_operations
                .iter()
                .flat_map(|operation| &operation.targets)
                .any(|target| {
                    target
                        .target_before
                        .as_ref()
                        .is_some_and(|before| before.oid == *sequence_oid)
                        && target.target_after.is_none()
                });
            if !exists_after && !dropped_by_lifecycle {
                return Err(EngineError::Durability(format!(
                    "transaction WAL advances unknown stable sequence identity {sequence_oid}"
                )));
            }
        }
        let mutation_tables = mutations
            .iter()
            .map(|mutation| match mutation {
                BinaryTransactionMutation::Insert { table, .. }
                | BinaryTransactionMutation::Update { table, .. }
                | BinaryTransactionMutation::Delete { table, .. } => table.as_str(),
            })
            .collect::<BTreeSet<_>>();
        let identity_tables = table_identities
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let expected_identity_tables = if ordered_envelope {
            ordered_row_names
                .iter()
                .filter(|table| !created_ordinals.contains_key(table.as_str()))
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
        } else {
            mutation_tables.clone()
        };
        if (ordered_envelope || !table_identities.is_empty())
            && expected_identity_tables != identity_tables
        {
            return Err(EngineError::Durability(
                "identity-bound transaction WAL does not cover its exact ordered row table set"
                    .to_string(),
            ));
        }
        for (table_name, identity) in &table_identities {
            let table = next_catalog
                .relational_catalog
                .get(table_name)
                .ok_or_else(|| {
                    EngineError::Durability(format!(
                        "identity-bound transaction targets unknown relation \"{table_name}\""
                    ))
                })?;
            let schema_digest = table_schema_digest(table)
                .map_err(|error| EngineError::Durability(error.to_string()))?;
            if table.oid != identity.table_oid || schema_digest != identity.schema_digest {
                return Err(EngineError::Durability(format!(
                    "identity-bound transaction relation \"{table_name}\" changed from OID {} before apply",
                    identity.table_oid
                )));
            }
        }
        if ordered_envelope {
            let reset_output_names = table_resets
                .iter()
                .map(|reset| reset.table.clone())
                .collect::<BTreeSet<_>>();
            if ordered_reset_names != reset_output_names
                || !mutation_tables
                    .iter()
                    .all(|table| ordered_row_names.contains(*table))
            {
                return Err(EngineError::Durability(
                    "ordered transaction outputs do not close over their statement identities"
                        .to_string(),
                ));
            }
            for reset in &table_resets {
                if usize::try_from(reset.ordinal)
                    .ok()
                    .and_then(|ordinal| operation_order.get(ordinal))
                    .is_none_or(|operation| {
                        !matches!(operation, BinaryTransactionOperationIdentity::TableReset { table } if table == &reset.table)
                    })
                {
                    return Err(EngineError::Durability(format!(
                        "transaction reset output for \"{}\" lost its statement position",
                        reset.table
                    )));
                }
            }
            for mutation in &mutations {
                let table = match mutation {
                    BinaryTransactionMutation::Insert { table, .. }
                    | BinaryTransactionMutation::Update { table, .. }
                    | BinaryTransactionMutation::Delete { table, .. } => table,
                };
                let last_reset = operation_order.iter().rposition(
                    |operation| matches!(operation, BinaryTransactionOperationIdentity::TableReset { table: reset_table } if reset_table == table),
                );
                if !operation_order
                    .iter()
                    .enumerate()
                    .any(|(ordinal, operation)| {
                        operation.matches_mutation(mutation)
                            && last_reset.is_none_or(|reset_ordinal| ordinal > reset_ordinal)
                    })
                {
                    return Err(EngineError::Durability(format!(
                        "resolved mutation for \"{table}\" has no row operation after its last reset"
                    )));
                }
            }
        }
        let mut validated_resets = Vec::with_capacity(table_resets.len());
        for reset in table_resets {
            let table = next_catalog
                .relational_catalog
                .get(&reset.table)
                .ok_or_else(|| {
                    EngineError::Durability(format!(
                        "transaction table reset targets unknown relation \"{}\"",
                        reset.table
                    ))
                })?
                .clone();
            let schema_digest = table_schema_digest(&table)
                .map_err(|error| EngineError::Durability(error.to_string()))?;
            if table.oid != reset.table_oid || schema_digest != reset.schema_digest {
                return Err(EngineError::Durability(format!(
                    "transaction table reset catalog identity mismatch for \"{}\"",
                    reset.table
                )));
            }
            if let Some((child, constraint)) =
                next_catalog.relational_catalog.values().find_map(|child| {
                    child
                        .foreign_keys
                        .iter()
                        .find(|foreign_key| foreign_key.referenced_table == table.name)
                        .map(|foreign_key| (child, foreign_key))
                })
            {
                return Err(EngineError::Durability(format!(
                    "transaction table reset for \"{}\" is not closed over inbound constraint \"{}\" on relation \"{}\"",
                    table.name, constraint.name, child.name
                )));
            }
            let expected_dependencies =
                table_access_dependency_identities(&next_catalog.relational_catalog, &table)
                    .map_err(|error| EngineError::Durability(error.to_string()))?;
            if expected_dependencies != reset.dependency_identities {
                return Err(EngineError::Durability(format!(
                    "transaction table reset dependency closure mismatch for \"{}\"",
                    reset.table
                )));
            }
            if table_reset_empty_digest(table.oid, schema_digest) != reset.after_empty_digest {
                return Err(EngineError::Durability(format!(
                    "transaction table reset empty-root digest mismatch for \"{}\"",
                    reset.table
                )));
            }
            let (visible_rows, before_digest) = self
                .table_reset_device_root_proof(
                    &table,
                    reset.source_commit_seq,
                    entry.index.saturating_sub(1),
                )
                .map_err(|error| EngineError::Durability(error.to_string()))?;
            if visible_rows != reset.expected_rows || before_digest != reset.before_digest {
                return Err(EngineError::Durability(format!(
                    "transaction table reset before-root proof mismatch for \"{}\"",
                    reset.table
                )));
            }
            validated_resets.push(reset);
        }
        // Early v1 row-transaction WAL could contain statement-order intermediate versions. The
        // current claimant writes a coalesced record, but replay must preserve those durable bytes.
        // Normalize both forms to the same final entity mutations before decode/publication so the
        // table-batched canonical path remains the sole device publisher.
        let mutations = Self::coalesce_transaction_mutations(mutations).map_err(|error| {
            EngineError::Durability(format!(
                "transaction WAL version chain could not be coalesced: {error}"
            ))
        })?;
        let reset_tables = validated_resets
            .iter()
            .map(|reset| reset.table.as_str())
            .collect::<BTreeSet<_>>();
        if let Some(table) = mutations.iter().find_map(|mutation| match mutation {
            BinaryTransactionMutation::Insert { .. } => None,
            BinaryTransactionMutation::Update { table, .. }
            | BinaryTransactionMutation::Delete { table, .. }
                if reset_tables.contains(table.as_str()) =>
            {
                Some(table)
            }
            _ => None,
        }) {
            return Err(EngineError::Durability(format!(
                "transaction table reset has a non-insert post-reset mutation for \"{table}\""
            )));
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
            let table = next_catalog
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

        // The full catalog+row decode has succeeded. Install the catalog working generation once;
        // row publication remains private to the enclosing commit until its one visibility join.
        *cat = next_catalog;
        let mut applied = Vec::with_capacity(validated_resets.len() + decoded.len());
        let mut device_authoritative_commits = validated_resets.len() as u64;
        let mut class_skips = 0u64;

        for reset in validated_resets {
            let mut write_set = WriteSet::default();
            write_set
                .tables
                .extend(reset.dependency_identities.keys().cloned());
            applied.push(AppliedRowMutation::TableReset { reset, write_set });
        }

        for mutation in decoded {
            let table = mutation.table().clone();
            let table_name = table.name.clone();
            let is_class = self.table_chunk_authoritative(&table_name).is_some();
            if is_class {
                class_skips = class_skips.saturating_add(1);
            } else {
                self.set_table_device_authoritative(&table_name, true);
                device_authoritative_commits = device_authoritative_commits.saturating_add(1);
            }
            match mutation {
                DecodedTransactionMutation::Insert { row_id, row, .. } => {
                    let mut write_set = WriteSet::default();
                    write_set.tables.insert(table_name.clone());
                    write_set.add_unique_slots(&table, &row);
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
                    write_set.tables.insert(table_name.clone());
                    write_set.rows.push(RowWriteKey {
                        table: table_name.clone(),
                        row_key: row_key.clone(),
                    });
                    write_set.add_unique_slots(&table, &old_row);
                    write_set.add_unique_slots(&table, &new_row);
                    let mut deduplicated = WriteSet::default();
                    deduplicated.extend_deduplicated(&write_set);
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
                    write_set.tables.insert(table_name.clone());
                    write_set.rows.push(RowWriteKey {
                        table: table_name.clone(),
                        row_key: row_key.clone(),
                    });
                    write_set.add_unique_slots(&table, &old_row);
                    applied.push(AppliedRowMutation::Delete {
                        table: table_name,
                        rows: vec![old_row],
                        write_set,
                        class_stamp: is_class.then(|| (vec![row_id], 0)),
                    });
                }
            }
        }

        // Nothing below can reject the record. Perform monotone/idempotent control-plane
        // allocator and sequence assignments; the enclosing commit publishes the device changes.
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
        for (sequence_oid, (last_value, is_called)) in sequence_advances_by_oid {
            // A value may precede DROP in the same ordered lifecycle. In that case the stable
            // identity is intentionally absent from the final catalog and its private state dies
            // with the object. Rename/recreate cannot redirect it because lookup is by OID.
            let Some(sequence) = cat
                .relational_sequences
                .values_mut()
                .find(|sequence| sequence.oid == sequence_oid)
            else {
                continue;
            };
            sequence.last_value = last_value;
            sequence.is_called = is_called;
        }
        self.read_state
            .residency
            .device_authoritative_commits
            .fetch_add(device_authoritative_commits, AtomicOrdering::Relaxed);
        self.read_state
            .residency
            .chunk_class_device_commits
            .fetch_add(class_skips, AtomicOrdering::Relaxed);
        Ok(applied)
    }
}
