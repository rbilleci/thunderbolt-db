//! Statement-snapshot ownership for explicit and predeclared transactions.
//!
//! READ COMMITTED rebases the immutable private GPU overlay over one newly captured publication
//! per statement. REPEATABLE READ performs the same capture lazily for its first data/catalog
//! statement, then retains it. The stable statement lock and delta Arc survive base replacement.

use super::*;
use crate::engine_mutation_admission::validate_transaction_characteristics;
use crate::engine_transaction_reset::{final_transaction_write_set, mutation_table};

impl Engine {
    /// Replace only the characteristics of an explicit transaction that has not acquired a
    /// statement snapshot or staged a mutation. The transaction identity, retained generation,
    /// private-delta owner, and active-snapshot accounting stay unchanged. Validation and every
    /// fallible precondition run before the keyed snapshot is replaced, so a rejected
    /// `SET TRANSACTION` cannot silently end the caller's transaction.
    pub(crate) fn set_empty_transaction_characteristics(
        &self,
        txn_id: TxnId,
        characteristics: TransactionCharacteristics,
    ) -> Result<TransactionCharacteristics, ExecuteError> {
        let characteristics = validate_transaction_characteristics(characteristics)?;
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        let current = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        self.ensure_transaction_not_program_owned(txn_id, &current)?;
        let statement_lock = Arc::clone(&current.statement_lock);
        let _statement = statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.ensure_transaction_snapshot_current(txn_id, &current)?;
        if current.data_snapshot_acquired.load(AtomicOrdering::Acquire)
            || !current.transaction_delta_is_empty()
        {
            return Err(ExecuteError::Unsupported(
                "SET TRANSACTION must precede the first transaction statement".to_string(),
            ));
        }

        let replacement = Arc::new(TransactionSnapshot {
            characteristics,
            boundary: current.boundary,
            next_row_id: current.next_row_id,
            catalog: Arc::clone(&current.catalog),
            table_versions: current.table_versions.clone(),
            resident_snapshots: Arc::clone(&current.resident_snapshots),
            resident_shards: Arc::clone(&current.resident_shards),
            device_authoritative_tables: Arc::clone(&current.device_authoritative_tables),
            chunk_authoritative_tables: Arc::clone(&current.chunk_authoritative_tables),
            delta: Arc::clone(&current.delta),
            table_access: Arc::clone(&current.table_access),
            rewrite_fenced_tables: Arc::clone(&current.rewrite_fenced_tables),
            statement_lock: Arc::clone(&current.statement_lock),
            program_owned: Arc::clone(&current.program_owned),
            data_snapshot_acquired: Arc::clone(&current.data_snapshot_acquired),
            base_streaming_cold_chunks: Arc::clone(&current.base_streaming_cold_chunks),
            private_gpu_account: Arc::clone(&current.private_gpu_account),
            _resident_index_resources: current._resident_index_resources.clone(),
            _resident_gpu_charge: Arc::clone(&current._resident_gpu_charge),
        });
        self.active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .replace_transaction_snapshot(txn_id, &current, replacement, false)?;
        Ok(characteristics)
    }

    pub(crate) fn refresh_transaction_snapshot_for_statement(
        &self,
        txn_id: TxnId,
        current: &Arc<TransactionSnapshot>,
    ) -> Result<Arc<TransactionSnapshot>, ExecuteError> {
        let isolation = match current.characteristics.isolation {
            TransactionIsolation::ReadUncommitted => TransactionIsolation::ReadCommitted,
            isolation => isolation,
        };
        let first_statement_snapshot =
            !current.data_snapshot_acquired.load(AtomicOrdering::Acquire);
        if isolation == TransactionIsolation::RepeatableRead
            && current.data_snapshot_acquired.load(AtomicOrdering::Acquire)
        {
            return Ok(Arc::clone(current));
        }

        // Capture one atomic publication. Kernel/rebase work runs after this lock is released; a
        // later commit may publish concurrently without changing the statement's chosen cut.
        let commit = self
            .commit_state_after_wave_quiescence()
            .map_err(ExecuteError::Engine)?;
        let fresh =
            self.capture_transaction_snapshot(self.committed_seq(), current.characteristics);
        drop(commit);

        if fresh.boundary == current.boundary {
            current
                .data_snapshot_acquired
                .store(true, AtomicOrdering::Release);
            return Ok(Arc::clone(current));
        }

        let rewrite_fenced_tables = self.rebase_transaction_delta(current, &fresh)?;
        let replacement = Arc::new(TransactionSnapshot {
            characteristics: current.characteristics,
            boundary: fresh.boundary,
            next_row_id: fresh.next_row_id,
            catalog: Arc::clone(&fresh.catalog),
            table_versions: fresh.table_versions.clone(),
            resident_snapshots: Arc::clone(&fresh.resident_snapshots),
            resident_shards: Arc::clone(&fresh.resident_shards),
            device_authoritative_tables: Arc::clone(&fresh.device_authoritative_tables),
            chunk_authoritative_tables: Arc::clone(&fresh.chunk_authoritative_tables),
            delta: Arc::clone(&current.delta),
            table_access: Arc::clone(&current.table_access),
            rewrite_fenced_tables,
            statement_lock: Arc::clone(&current.statement_lock),
            program_owned: Arc::clone(&current.program_owned),
            data_snapshot_acquired: Arc::clone(&current.data_snapshot_acquired),
            base_streaming_cold_chunks: Arc::clone(&fresh.base_streaming_cold_chunks),
            private_gpu_account: Arc::clone(&current.private_gpu_account),
            _resident_index_resources: fresh._resident_index_resources.clone(),
            _resident_gpu_charge: Arc::clone(&fresh._resident_gpu_charge),
        });
        replacement
            .data_snapshot_acquired
            .store(true, AtomicOrdering::Release);
        self.active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .replace_transaction_snapshot(
                txn_id,
                current,
                Arc::clone(&replacement),
                first_statement_snapshot,
            )?;
        Ok(replacement)
    }

    fn rebase_transaction_delta(
        &self,
        current: &Arc<TransactionSnapshot>,
        fresh: &Arc<TransactionSnapshot>,
    ) -> Result<Arc<Mutex<BTreeSet<String>>>, ExecuteError> {
        // Rewrite-fence membership is relative to one statement boundary. A READ COMMITTED rebase
        // has captured the publication containing every earlier fence, and already-held table
        // guards prevent a same-OID reset from crossing this replay. Start the fresh statement
        // unmarked; its ordinary access acquisition will bind any reset that races after capture.
        let rebased_rewrite_fenced_tables = Arc::new(Mutex::new(BTreeSet::new()));
        let (rebased_catalog_overlay, private_catalog_tables) = {
            let delta = current
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let catalog_commands = delta
                .operations
                .iter()
                .filter_map(|operation| match operation {
                    TransactionOperation::Catalog(staged) => Some(staged.as_ref()),
                    TransactionOperation::Row(_) | TransactionOperation::TableReset(_) => None,
                })
                .collect::<Vec<_>>();
            let has_sequence_resets = delta.operations.iter().any(|operation| {
                matches!(
                    operation,
                    TransactionOperation::TableReset(reset)
                        if reset.sequence_reset_identity.is_some()
                )
            });
            if !catalog_commands.is_empty() || has_sequence_resets {
                let base = delta.catalog_base.as_deref().ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "transactional catalog operation envelope lost its base generation"
                            .to_string(),
                    ))
                })?;
                if !base.same_contents(fresh.catalog.as_ref()) {
                    return Err(ExecuteError::Serialization(
                        "published catalog contents changed after transactional DDL staging"
                            .to_string(),
                    ));
                }
                let mut overlay = delta.catalog_overlay.as_deref().cloned().ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "transactional catalog operation envelope lost its private catalog overlay"
                            .to_string(),
                    ))
                })?;
                // Only the publication stamp changed. Object/column allocators and every catalog
                // dependency are byte-identical, so the private overlay retains its stable
                // identities and is rebound to the fresh READ COMMITTED statement boundary.
                overlay.commit_seq = fresh.catalog.commit_seq;
                let private_tables = catalog_commands
                    .into_iter()
                    .filter_map(|catalog_command| match &catalog_command.command {
                        Command::CreateTable(create) => Some(create.table.clone()),
                        Command::CreateView(_) | Command::RenameView(_) | Command::DropView(_) => {
                            None
                        }
                        Command::CreateIndex(_)
                        | Command::RenameIndex(_)
                        | Command::DropIndex(_) => None,
                        Command::CreateSequence(_)
                        | Command::SequenceRestart(_)
                        | Command::RenameSequence(_)
                        | Command::DropSequence(_) => None,
                        _ => {
                            unreachable!(
                                "transactional catalog staging admitted an unsupported family"
                            )
                        }
                    })
                    .collect();
                (Some(Arc::new(overlay)), private_tables)
            } else {
                (None, Vec::new())
            }
        };
        let transaction_catalog = rebased_catalog_overlay.as_ref().unwrap_or(&fresh.catalog);
        let (
            generation,
            mut operations,
            sequence_state,
            sequence_state_by_oid,
            old_private_gpu_bytes,
        ) = {
            let delta = current
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                delta.generation,
                delta.operations.clone(),
                delta.sequence_state.clone(),
                delta.sequence_state_by_oid.clone(),
                delta.private_gpu_bytes_by_gpu.clone(),
            )
        };
        rekey_provisional_inserts(&mut operations, fresh.next_row_id)?;

        let mut next_shards = (*fresh.resident_shards).clone();
        // A transaction-created table has no globally published shard generation. Rebase starts
        // it from the same private empty generation installed by CREATE, then replays the staged
        // row deltas in order; copying its prior final shard would double-apply those mutations.
        for table in private_catalog_tables {
            next_shards.entry(table).or_default();
        }
        let mut next_cold_chunks = (*fresh.base_streaming_cold_chunks).clone();
        // Replay is itself a tiny private transaction generation. Deep device locate/materialize
        // helpers resolve through the scoped snapshot, so each delta reads the exact captured
        // fresh base plus the replay prefix already installed in `next_shards`; they never consult
        // a newer global generation that may publish after capture.
        let scratch_shards = Arc::new(next_shards.clone());
        let scratch_cold_chunks = Arc::new(next_cold_chunks.clone());
        let scratch_delta = Arc::new(std::sync::Mutex::new(TransactionDeltaState {
            generation: 0,
            resident_shards: Arc::clone(&scratch_shards),
            resident_shards_authority: scratch_shards,
            streaming_cold_chunks: Arc::clone(&scratch_cold_chunks),
            streaming_cold_chunks_authority: scratch_cold_chunks,
            operations: Vec::new(),
            write_set: WriteSet::default(),
            next_row_id: fresh.next_row_id,
            sequence_state: BTreeMap::new(),
            sequence_state_by_oid: BTreeMap::new(),
            catalog_base: None,
            catalog_overlay: rebased_catalog_overlay.clone(),
            private_gpu_bytes_by_gpu: BTreeMap::new(),
            commit_gpu_bytes_by_gpu: BTreeMap::new(),
        }));
        let scratch = Arc::new(TransactionSnapshot {
            characteristics: fresh.characteristics,
            boundary: fresh.boundary,
            next_row_id: fresh.next_row_id,
            catalog: Arc::clone(&fresh.catalog),
            table_versions: fresh.table_versions.clone(),
            resident_snapshots: Arc::clone(&fresh.resident_snapshots),
            resident_shards: Arc::clone(&fresh.resident_shards),
            device_authoritative_tables: Arc::clone(&fresh.device_authoritative_tables),
            chunk_authoritative_tables: Arc::clone(&fresh.chunk_authoritative_tables),
            delta: Arc::clone(&scratch_delta),
            table_access: Arc::clone(&current.table_access),
            rewrite_fenced_tables: Arc::clone(&rebased_rewrite_fenced_tables),
            statement_lock: Arc::new(std::sync::Mutex::new(())),
            program_owned: Arc::new(AtomicBool::new(false)),
            data_snapshot_acquired: Arc::new(AtomicBool::new(true)),
            base_streaming_cold_chunks: Arc::clone(&fresh.base_streaming_cold_chunks),
            private_gpu_account: Arc::new(std::sync::Mutex::new(BTreeMap::new())),
            _resident_index_resources: fresh._resident_index_resources.clone(),
            _resident_gpu_charge: Arc::clone(&fresh._resident_gpu_charge),
        });
        let mut gpu_reservation = TransactionGpuReservation::new(self);
        let catalog_commands = operations
            .iter()
            .filter_map(|operation| match operation {
                TransactionOperation::Catalog(staged) => Some(staged.as_ref().clone()),
                TransactionOperation::Row(_) | TransactionOperation::TableReset(_) => None,
            })
            .collect::<Vec<_>>();
        for (ordinal, operation) in operations.iter().enumerate() {
            match operation {
                TransactionOperation::Catalog(staged) => {
                    if usize::try_from(staged.ordinal).ok() != Some(ordinal) {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "transactional catalog operation lost statement order during rebase"
                                .to_string(),
                        )));
                    }
                }
                TransactionOperation::Row(delta) => {
                    for (name, expected) in &delta.catalog_dependencies {
                        if transaction_catalog
                            .relational_catalog
                            .get(name)
                            .is_none_or(|observed| {
                                !same_transaction_row_catalog_dependency(
                                    expected,
                                    observed,
                                    &delta.sequence_input_oids,
                                    transaction_catalog,
                                    &catalog_commands,
                                )
                            })
                        {
                            return Err(ExecuteError::Serialization(format!(
                                "catalog dependency \"{name}\" changed after transaction statement snapshot {}",
                                delta.read_snapshot
                            )));
                        }
                    }
                    let table_name = mutation_table(&delta.mutation);
                    let table = transaction_catalog
                        .relational_catalog
                        .get(table_name)
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                                "relation \"{table_name}\" changed before the next READ COMMITTED statement"
                            ))
                        })?;
                    let _scope = self.enter_transaction_read(Arc::clone(&scratch));
                    self.apply_transaction_private_delta(
                        table,
                        delta,
                        fresh.boundary,
                        &mut next_shards,
                        &mut next_cold_chunks,
                        &mut gpu_reservation,
                    )
                    .map_err(read_committed_rebase_error)?;
                }
                TransactionOperation::TableReset(reset) => {
                    if usize::try_from(reset.ordinal).ok() != Some(ordinal) {
                        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                            "transaction table reset lost statement order during rebase"
                                .to_string(),
                        )));
                    }
                    for (name, expected) in &reset.catalog_dependencies {
                        if transaction_catalog.relational_catalog.get(name) != Some(expected) {
                            return Err(ExecuteError::Serialization(format!(
                                "table reset catalog dependency \"{name}\" changed after snapshot {}",
                                reset.read_snapshot
                            )));
                        }
                    }
                    let table = transaction_catalog
                        .relational_catalog
                        .get(&reset.table)
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                                "relation \"{}\" changed before its table reset was rebased",
                                reset.table
                            ))
                        })?;
                    next_shards.insert(reset.table.clone(), Vec::new());
                    next_cold_chunks.remove(&reset.table);
                    self.append_transaction_empty_root(
                        table,
                        &mut next_shards,
                        &mut gpu_reservation,
                    )
                    .map_err(read_committed_rebase_error)?;
                }
            }
            let mut replay = scratch_delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            replay.publish_resident_shards(Arc::new(next_shards.clone()));
            replay.publish_streaming_cold_chunks(Arc::new(next_cold_chunks.clone()));
            replay.operations.push(operation.clone());
            replay.write_set = final_transaction_write_set(&replay.operations);
            replay.generation = replay.generation.saturating_add(1);
        }
        let next_private_gpu_bytes =
            transaction_private_shard_bytes(fresh.resident_shards.as_ref(), &next_shards);
        gpu_reservation
            .ensure_replacement_admitted(&old_private_gpu_bytes, &next_private_gpu_bytes)?;

        let rows_consumed = operations.iter().try_fold(0u64, |total, operation| {
            let consumed = match operation {
                TransactionOperation::Catalog(_) => 0,
                TransactionOperation::Row(delta) => delta.rows_consumed,
                TransactionOperation::TableReset(_) => 0,
            };
            total.checked_add(consumed).ok_or_else(|| {
                ExecuteError::Unsupported(
                    "transaction provisional row identity count overflow".to_string(),
                )
            })
        })?;
        let mut state = current
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.generation != generation {
            return Err(ExecuteError::Serialization(
                "transaction changed while its READ COMMITTED statement snapshot was rebased"
                    .to_string(),
            ));
        }
        state.publish_resident_shards(Arc::new(next_shards));
        state.publish_streaming_cold_chunks(Arc::new(next_cold_chunks));
        state.operations = operations;
        state.write_set = final_transaction_write_set(&state.operations);
        state.next_row_id = fresh
            .next_row_id
            .checked_add(rows_consumed)
            .ok_or_else(|| {
                ExecuteError::Unsupported(
                    "transaction provisional row identity space exhausted".to_string(),
                )
            })?;
        state.sequence_state = sequence_state;
        state.sequence_state_by_oid = sequence_state_by_oid;
        if let Some(overlay) = rebased_catalog_overlay {
            state.catalog_base = Some(Arc::clone(&fresh.catalog));
            state.catalog_overlay = Some(overlay);
        }
        gpu_reservation
            .replace_charges(&mut state.private_gpu_bytes_by_gpu, next_private_gpu_bytes);
        Ok(rebased_rewrite_fenced_tables)
    }
}

fn read_committed_rebase_error(error: ExecuteError) -> ExecuteError {
    match error {
        ExecuteError::Serialization(_) => error,
        ExecuteError::Engine(EngineError::ApplyFailed(message)) => ExecuteError::Serialization(
            format!("READ COMMITTED rebase found a stale transaction dependency: {message}"),
        ),
        other => other,
    }
}

fn rekey_provisional_inserts(
    operations: &mut [TransactionOperation],
    new_base: u64,
) -> Result<(), ExecuteError> {
    let mut mapping = BTreeMap::<(String, u64), u64>::new();
    let mut next = new_base;
    for delta in operations.iter().filter_map(|operation| match operation {
        TransactionOperation::Row(delta) => Some(delta),
        TransactionOperation::Catalog(_) | TransactionOperation::TableReset(_) => None,
    }) {
        let PreparedMutation::Insert {
            table,
            inserted_rows,
            ..
        } = &delta.mutation
        else {
            continue;
        };
        let prefix = relational_key_prefix(table);
        for (key, _) in inserted_rows {
            let old = crate::engine_residency::parse_relational_row_id(key, &prefix).ok_or_else(
                || {
                    ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction INSERT lost provisional identity during rebase".to_string(),
                    ))
                },
            )?;
            mapping.entry((table.clone(), old)).or_insert_with(|| {
                let assigned = next;
                next = next.saturating_add(1);
                assigned
            });
        }
    }
    if next == u64::MAX && !mapping.is_empty() {
        return Err(ExecuteError::Unsupported(
            "transaction provisional row identity space exhausted".to_string(),
        ));
    }

    for delta in operations
        .iter_mut()
        .filter_map(|operation| match operation {
            TransactionOperation::Row(delta) => Some(Arc::make_mut(delta)),
            TransactionOperation::Catalog(_) | TransactionOperation::TableReset(_) => None,
        })
    {
        let table = match &delta.mutation {
            PreparedMutation::Insert { table, .. }
            | PreparedMutation::Update { table, .. }
            | PreparedMutation::Delete { table, .. } => table.clone(),
        };
        let prefix = relational_key_prefix(&table);
        let rewrite_key = |key: &mut String| {
            if let Some(old) = crate::engine_residency::parse_relational_row_id(key, &prefix) {
                if let Some(new) = mapping.get(&(table.clone(), old)) {
                    *key = relational_row_key(&table, *new);
                }
            }
        };
        match &mut delta.mutation {
            PreparedMutation::Insert { inserted_rows, .. } => {
                for (key, _) in inserted_rows {
                    rewrite_key(key);
                }
            }
            PreparedMutation::Update { installs, .. } => {
                for (tuple_id, key, _) in installs {
                    if let Some(new) = mapping.get(&(table.clone(), *tuple_id)) {
                        *tuple_id = *new;
                    }
                    rewrite_key(key);
                }
            }
            PreparedMutation::Delete {
                tuple_ids,
                class_epoch,
                ..
            } => {
                if class_epoch.is_none() {
                    for tuple_id in tuple_ids {
                        if let Some(new) = mapping.get(&(table.clone(), *tuple_id)) {
                            *tuple_id = *new;
                        }
                    }
                }
            }
        }
        for row in &mut delta.write_set.rows {
            if row.table == table {
                rewrite_key(&mut row.row_key);
            }
        }
    }
    Ok(())
}
