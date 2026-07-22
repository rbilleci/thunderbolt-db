//! Statement-snapshot ownership for explicit and predeclared transactions.
//!
//! READ COMMITTED rebases the immutable private GPU overlay over one newly captured publication
//! per statement. REPEATABLE READ performs the same capture lazily for its first data/catalog
//! statement, then retains it. The stable statement lock and delta Arc survive base replacement.

use super::*;
use crate::engine_mutation_admission::validate_transaction_characteristics;

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

        self.rebase_transaction_delta(current, &fresh)?;
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
    ) -> Result<(), ExecuteError> {
        {
            let delta = current
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if delta.catalog_command.is_some()
                && delta.catalog_base.as_deref() != Some(fresh.catalog.as_ref())
            {
                return Err(ExecuteError::Serialization(
                    "published state changed after transactional DDL staging".to_string(),
                ));
            }
        }
        let (generation, mut deltas, write_set, sequence_state, old_private_gpu_bytes) = {
            let delta = current
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                delta.generation,
                delta.deltas.clone(),
                delta.write_set.clone(),
                delta.sequence_state.clone(),
                delta.private_gpu_bytes_by_gpu.clone(),
            )
        };
        rekey_provisional_inserts(&mut deltas, fresh.next_row_id)?;

        let mut next_shards = (*fresh.resident_shards).clone();
        let mut next_cold_chunks = (*fresh.base_streaming_cold_chunks).clone();
        // Replay is itself a tiny private transaction generation. Deep device locate/materialize
        // helpers resolve through the scoped snapshot, so each delta reads the exact captured
        // fresh base plus the replay prefix already installed in `next_shards`; they never consult
        // a newer global generation that may publish after capture.
        let scratch_delta = Arc::new(std::sync::Mutex::new(TransactionDeltaState {
            generation: 0,
            resident_shards: Arc::new(next_shards.clone()),
            streaming_cold_chunks: Arc::new(next_cold_chunks.clone()),
            deltas: Vec::new(),
            write_set: WriteSet::default(),
            next_row_id: fresh.next_row_id,
            sequence_state: BTreeMap::new(),
            catalog_command: None,
            catalog_base: None,
            catalog_overlay: None,
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
            statement_lock: Arc::new(std::sync::Mutex::new(())),
            program_owned: Arc::new(AtomicBool::new(false)),
            data_snapshot_acquired: Arc::new(AtomicBool::new(true)),
            base_streaming_cold_chunks: Arc::clone(&fresh.base_streaming_cold_chunks),
            private_gpu_account: Arc::new(std::sync::Mutex::new(BTreeMap::new())),
            _resident_index_resources: fresh._resident_index_resources.clone(),
            _resident_gpu_charge: Arc::clone(&fresh._resident_gpu_charge),
        });
        let mut gpu_reservation = TransactionGpuReservation::new(self);
        for delta in &deltas {
            for (name, expected) in &delta.catalog_dependencies {
                if fresh.catalog.relational_catalog.get(name) != Some(expected) {
                    return Err(ExecuteError::Serialization(format!(
                        "catalog dependency \"{name}\" changed after transaction statement snapshot {}",
                        delta.read_snapshot
                    )));
                }
            }
            let table_name = match &delta.mutation {
                PreparedMutation::Insert { table, .. }
                | PreparedMutation::Update { table, .. }
                | PreparedMutation::Delete { table, .. } => table,
            };
            let table = fresh
                .catalog
                .relational_catalog
                .get(table_name)
                .ok_or_else(|| {
                    ExecuteError::Serialization(format!(
                        "relation \"{table_name}\" changed before the next READ COMMITTED statement"
                    ))
                })?;
            {
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
            let mut replay = scratch_delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            replay.resident_shards = Arc::new(next_shards.clone());
            replay.streaming_cold_chunks = Arc::new(next_cold_chunks.clone());
            replay.deltas.push(delta.clone());
            replay.write_set.extend_deduplicated(&delta.write_set);
            replay.generation = replay.generation.saturating_add(1);
        }
        let next_private_gpu_bytes =
            transaction_private_shard_bytes(fresh.resident_shards.as_ref(), &next_shards);
        gpu_reservation
            .ensure_replacement_admitted(&old_private_gpu_bytes, &next_private_gpu_bytes)?;

        let rows_consumed = deltas.iter().try_fold(0u64, |total, delta| {
            total.checked_add(delta.rows_consumed).ok_or_else(|| {
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
        state.resident_shards = Arc::new(next_shards);
        state.streaming_cold_chunks = Arc::new(next_cold_chunks);
        state.deltas = deltas;
        state.write_set = write_set;
        state.next_row_id = fresh
            .next_row_id
            .checked_add(rows_consumed)
            .ok_or_else(|| {
                ExecuteError::Unsupported(
                    "transaction provisional row identity space exhausted".to_string(),
                )
            })?;
        state.sequence_state = sequence_state;
        gpu_reservation
            .replace_charges(&mut state.private_gpu_bytes_by_gpu, next_private_gpu_bytes);
        Ok(())
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

fn rekey_provisional_inserts(deltas: &mut [WriteDelta], new_base: u64) -> Result<(), ExecuteError> {
    let mut mapping = BTreeMap::<(String, u64), u64>::new();
    let mut next = new_base;
    for delta in deltas.iter() {
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

    for delta in deltas {
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
