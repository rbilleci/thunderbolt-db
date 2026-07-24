//! Explicit-transaction private GPU generation: stage prepared DML into immutable shard overlays
//! without mutating the globally published relation. The transaction snapshot remains the owner;
//! SELECT and subsequent DML load the latest private generation through the ordinary residency
//! accessors.

use super::*;
use crate::engine_transaction_catalog::TransactionCatalogEnvelopeSlices;
use crate::engine_transaction_reset::{
    final_transaction_operations, final_transaction_row_operations, final_transaction_write_set,
    table_access_dependency_identities, table_schema_digest, transaction_row_deltas,
};

pub(crate) mod gpu_accounting;
mod isolation;
mod ordered_wal;
mod record;
use gpu_accounting::transaction_private_shard_bytes;
pub(crate) use gpu_accounting::TransactionGpuReservation;

type TargetRow<'a> = (u64, &'a [SqlValue]);

impl Engine {
    #[cfg(test)]
    pub(crate) fn fail_next_transaction_post_durable_apply(&self) {
        self.fail_next_transaction_post_durable_apply
            .store(true, AtomicOrdering::Release);
    }

    #[cfg(test)]
    pub(crate) fn set_transaction_post_durable_hook(&self, hook: impl FnOnce() + Send + 'static) {
        *self
            .transaction_post_durable_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Box::new(hook));
    }

    /// Drive COMMIT and an optional successor as one engine-owned transaction-control operation.
    /// Returning the successor identity lets protocol façades adopt the transaction that was
    /// registered under the same commit/active-snapshot locks, preserving `AND CHAIN` atomicity.
    pub fn commit_explicit_transaction(
        &self,
        txn_id: TxnId,
        chain: bool,
    ) -> Result<Option<TxnId>, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if let Some(snapshot) = self.transaction_snapshot_handle(txn_id) {
            self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
            let _statement = snapshot
                .statement_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
            self.commit_explicit_transaction_statement_locked(txn_id, chain, &snapshot)
        } else {
            self.finish_transaction_context(txn_id, true, chain)
                .map_err(ExecuteError::Txn)
        }
    }

    pub(crate) fn commit_explicit_transaction_statement_locked(
        &self,
        txn_id: TxnId,
        chain: bool,
        snapshot: &Arc<TransactionSnapshot>,
    ) -> Result<Option<TxnId>, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        self.ensure_transaction_snapshot_current(txn_id, snapshot)?;
        if !snapshot.transaction_delta_is_empty() {
            self.commit_transaction_delta(txn_id, chain, current_timestamp_micros())
        } else {
            self.finish_transaction_context(txn_id, true, chain)
                .map_err(ExecuteError::Txn)
        }
    }

    pub fn rollback_explicit_transaction(
        &self,
        txn_id: TxnId,
        chain: bool,
    ) -> Result<Option<TxnId>, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if let Some(snapshot) = self.transaction_snapshot_handle(txn_id) {
            self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
            let _statement = snapshot
                .statement_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
            self.rollback_explicit_transaction_statement_locked(txn_id, chain, &snapshot)
        } else {
            self.finish_transaction_context(txn_id, false, chain)
                .map_err(ExecuteError::Txn)
        }
    }

    pub(crate) fn rollback_explicit_transaction_statement_locked(
        &self,
        txn_id: TxnId,
        chain: bool,
        snapshot: &Arc<TransactionSnapshot>,
    ) -> Result<Option<TxnId>, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        self.ensure_transaction_snapshot_current(txn_id, snapshot)?;
        self.finish_transaction_context(txn_id, false, chain)
            .map_err(ExecuteError::Txn)
    }

    /// Stage one DML statement in an explicit transaction. Preparation and every constraint/read
    /// bind to the retained generation plus prior private deltas. A successful statement publishes
    /// a new transaction-private shard map atomically; global WAL, MVCC, residency, and committed
    /// visibility remain untouched until COMMIT.
    pub fn execute_dml_in_transaction(
        &self,
        txn_id: TxnId,
        text: &str,
    ) -> Result<(), ExecuteError> {
        let command = parse_command(text)?;
        if crate::engine_dml_concurrent::command_has_returning(&command) {
            return Err(crate::engine_dml_concurrent::discarded_returning_error());
        }
        self.execute_parsed_dml_in_transaction_with_result(txn_id, command)
            .map(|_| ())
    }

    pub fn execute_dml_in_transaction_with_result(
        &self,
        txn_id: TxnId,
        text: &str,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        let command = parse_command(text)?;
        self.execute_parsed_dml_in_transaction_with_result(txn_id, command)
    }

    pub(crate) fn execute_parsed_dml_in_transaction_with_result(
        &self,
        txn_id: TxnId,
        command: Command,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        let snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        self.ensure_transaction_not_program_owned(txn_id, &snapshot)?;
        let statement_lock = Arc::clone(&snapshot.statement_lock);
        let _statement = statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
        let snapshot = self.refresh_transaction_snapshot_for_statement(txn_id, &snapshot)?;
        self.execute_parsed_dml_in_transaction_statement_locked(txn_id, command, &snapshot)
    }

    pub(crate) fn execute_parsed_dml_in_transaction_statement_locked(
        &self,
        txn_id: TxnId,
        command: Command,
        snapshot: &Arc<TransactionSnapshot>,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        self.execute_parsed_dml_in_transaction_statement_locked_with_sequence_parent(
            txn_id, command, snapshot, false,
        )
    }

    pub(crate) fn execute_parsed_dml_in_transaction_statement_locked_with_sequence_parent(
        &self,
        txn_id: TxnId,
        mut command: Command,
        snapshot: &Arc<TransactionSnapshot>,
        sequence_parent_autocommit: bool,
    ) -> Result<DmlExecutionResult, ExecuteError> {
        self.legacy_lane_history_write_guard()
            .map_err(ExecuteError::Engine)?;
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        self.ensure_transaction_snapshot_current(txn_id, snapshot)?;
        if !matches!(
            &command,
            Command::Insert(_) | Command::Update(_) | Command::Delete(_)
        ) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "explicit transaction staging accepts INSERT, UPDATE, or DELETE".to_string(),
            )));
        }
        if snapshot.characteristics.access == TransactionAccessMode::ReadOnly {
            return Err(ExecuteError::Unsupported(
                "cannot execute DML in a READ ONLY transaction".to_string(),
            ));
        }
        let statement_digest = transaction_statement_digest(&command)?;

        let table_name = match &command {
            Command::Insert(insert) => insert.table.clone(),
            Command::Update(update) => update.table.clone(),
            Command::Delete(delete) => delete.table.clone(),
            _ => unreachable!("DML shape checked above"),
        };
        let transaction_catalog = snapshot.transaction_catalog();
        let table = transaction_catalog
            .relational_catalog
            .get(&table_name)
            .cloned()
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{table_name}\" does not exist in the transaction generation"
                )))
            })?;
        let access_identities =
            table_access_dependency_identities(&transaction_catalog.relational_catalog, &table)?;
        // RR reads retain their historical catalog/OID binding, but writes may not target an
        // object that has since been renamed, dropped, or recreated under the same name.
        self.acquire_transaction_write_table_access_identities(snapshot, &access_identities)?;
        let sequence_access_identities = table
            .columns
            .iter()
            .filter_map(|column| match &column.default {
                Some(ColumnDefault::SequenceNextVal { sequence, .. }) => transaction_catalog
                    .relational_sequences
                    .get(sequence)
                    .map(|sequence| sequence.oid),
                None | Some(ColumnDefault::Literal(_)) => None,
            })
            .collect::<BTreeSet<_>>();
        snapshot
            .table_access
            .acquire_shared(sequence_access_identities)?;

        let (generation, next_row_id, statement_ordinal, expression_ordinal_base) = {
            let delta = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                delta.generation,
                delta.next_row_id,
                u32::try_from(delta.operations.len()).map_err(|_| {
                    ExecuteError::Unsupported(
                        "transaction operation count exceeds typed WAL framing".to_string(),
                    )
                })?,
                u32::try_from(
                    delta
                        .sequence_value_references
                        .iter()
                        .filter(|reference| {
                            usize::try_from(reference.statement_ordinal).ok()
                                == Some(delta.operations.len())
                        })
                        .count(),
                )
                .map_err(|_| {
                    ExecuteError::Unsupported(
                        "transaction sequence expression count exceeds typed WAL framing"
                            .to_string(),
                    )
                })?,
            )
        };
        let mut sequence_value_references = self.materialize_published_sequence_defaults(
            snapshot,
            &transaction_catalog,
            &table,
            &mut command,
            crate::engine_sequence_value::SequenceDefaultStatementIdentity {
                parent_txn_id: txn_id,
                parent_autocommit: sequence_parent_autocommit,
                statement_ordinal,
                expression_ordinal_base,
                parent_request_digest: statement_digest,
            },
        )?;
        let _scope = self.enter_transaction_read(Arc::clone(snapshot));
        let dml_snapshot = DmlReadSnapshot {
            commit_seq: snapshot.boundary,
            next_row_id,
        };
        let prepared = self.prepare_dml(&command, dml_snapshot, InsertPrepareValidation::Full)?;
        self.bind_sequence_default_insert_rows(&table, &prepared, &mut sequence_value_references)?;
        let prepared_sequence_state = match &prepared.mutation {
            PreparedMutation::Insert { seq_advances, .. } => seq_advances.clone(),
            PreparedMutation::Update { .. } | PreparedMutation::Delete { .. } => BTreeMap::new(),
        };
        let sequence_input_oids = prepared_sequence_state
            .keys()
            .map(|name| {
                transaction_catalog
                    .relational_sequences
                    .get(name)
                    .map(|sequence| (name.clone(), sequence.oid))
                    .ok_or_else(|| {
                        ExecuteError::Serialization(format!(
                            "transaction sequence input \"{name}\" left its private catalog generation"
                        ))
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let prepared_sequence_state_by_oid = prepared_sequence_state
            .iter()
            .map(|(name, state)| {
                (
                    *sequence_input_oids
                        .get(name)
                        .expect("captured every prepared sequence input"),
                    *state,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let rows_affected = prepared.rows_affected();
        let returning = self.project_dml_returning(&command, &prepared, snapshot.boundary)?;

        self.validate_transaction_delta_residency(
            &table,
            snapshot
                .catalog
                .relational_catalog
                .contains_key(&table_name),
        )?;

        let current_shards = snapshot.transaction_shards();
        let current_cold_chunks = snapshot.transaction_cold_chunks();
        let mut next_shards = (*current_shards).clone();
        let mut next_cold_chunks = (*current_cold_chunks).clone();
        let mut gpu_reservation = TransactionGpuReservation::new(self);
        self.apply_transaction_private_delta(
            &table,
            &prepared,
            snapshot.boundary,
            &mut next_shards,
            &mut next_cold_chunks,
            &mut gpu_reservation,
        )?;
        let next_private_gpu_bytes =
            transaction_private_shard_bytes(snapshot.resident_shards.as_ref(), &next_shards);

        // Compare-and-publish under the transaction mutex. Connection/session execution is ordered,
        // but the generation check also makes accidental same-transaction concurrent use fail loud
        // instead of losing one private statement.
        let mut delta = snapshot
            .delta
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if delta.generation != generation {
            return Err(ExecuteError::Serialization(
                "concurrent statements attempted to publish the same transaction delta".to_string(),
            ));
        }
        gpu_reservation.ensure_replacement_admitted(
            &delta.private_gpu_bytes_by_gpu,
            &next_private_gpu_bytes,
        )?;
        u32::try_from(delta.operations.len()).map_err(|_| {
            ExecuteError::Unsupported(
                "transaction operation count exceeds typed WAL framing".to_string(),
            )
        })?;
        delta.next_row_id = delta.next_row_id.saturating_add(prepared.rows_consumed);
        delta.sequence_state.extend(prepared_sequence_state);
        delta
            .sequence_state_by_oid
            .extend(prepared_sequence_state_by_oid);
        delta
            .sequence_value_references
            .extend(sequence_value_references);
        delta
            .operations
            .push(TransactionOperation::Row(Arc::new(StagedRowOperation {
                statement_digest,
                sequence_input_oids,
                delta: prepared,
            })));
        delta.write_set = final_transaction_write_set(&delta.operations);
        delta.publish_resident_shards(Arc::new(next_shards));
        delta.publish_streaming_cold_chunks(Arc::new(next_cold_chunks));
        delta.generation = delta.generation.saturating_add(1);
        // The transaction statement lock excludes every other same-transaction reader or writer.
        // Release this statement's old-map pin before dropping superseded allocation charges, then
        // atomically convert the temporary reservations into the exact replacement-generation account.
        drop(current_shards);
        gpu_reservation
            .replace_charges(&mut delta.private_gpu_bytes_by_gpu, next_private_gpu_bytes);
        Ok(DmlExecutionResult {
            rows_affected,
            returning,
        })
    }

    /// Durably publish all staged statements as ONE resolved WAL record and ONE MVCC generation.
    /// The commit lock closes catalog/conflict races, final insert identities are claimed once, and
    /// recovery applies the same ordered row mutations without re-evaluating predicates.
    pub(crate) fn commit_transaction_delta(
        &self,
        txn_id: TxnId,
        chain: bool,
        timestamp_micros: u64,
    ) -> Result<Option<TxnId>, ExecuteError> {
        self.commit_transaction_delta_inner(txn_id, chain, timestamp_micros, None)
    }

    pub(crate) fn commit_claimed_transaction_delta(
        &self,
        txn_id: TxnId,
        request_digest: gpu_db_wal::CanonicalDigest,
        timestamp_micros: u64,
    ) -> Result<(), ExecuteError> {
        let snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        let _statement = snapshot
            .statement_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.ensure_transaction_snapshot_current(txn_id, &snapshot)?;
        self.commit_transaction_delta_inner(txn_id, false, timestamp_micros, Some(request_digest))?;
        Ok(())
    }

    fn commit_transaction_delta_inner(
        &self,
        txn_id: TxnId,
        chain: bool,
        timestamp_micros: u64,
        request_digest_override: Option<gpu_db_wal::CanonicalDigest>,
    ) -> Result<Option<TxnId>, ExecuteError> {
        self.ensure_commit_path_available()
            .map_err(ExecuteError::Engine)?;
        if self.repl_role() != Role::Leader {
            return Err(ExecuteError::Engine(EngineError::NotLeader));
        }
        let snapshot = self
            .transaction_snapshot_handle(txn_id)
            .ok_or(ExecuteError::Txn(TxnError::NotFound(txn_id)))?;
        // A classic wave installs its device generation before its durability tail publishes
        // `committed_seq`. Explicit-transaction row/unique/FK validation must not inspect that
        // applied-but-unpublished interval. Drain first, then prove under the commit lock that no
        // sequencer handed off a new tail in the acquisition gap; retry until the cut is settled.
        let mut commit = self
            .commit_state_after_wave_quiescence()
            .map_err(ExecuteError::Engine)?;
        self.legacy_lane_history_write_guard()
            .map_err(ExecuteError::Engine)?;
        if commit.txn_manager.state(txn_id) != Some(TxnState::Active) {
            return Err(ExecuteError::Txn(TxnError::NotActive(txn_id)));
        }
        if commit.transaction_status.contains_key(&txn_id) {
            return Err(ExecuteError::Engine(EngineError::Durability(format!(
                "transaction id {txn_id} was already claimed by another write strategy while the explicit transaction was active"
            ))));
        }
        let (operations, catalog_base, sequence_value_references) = {
            let delta = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                delta.operations.clone(),
                delta.catalog_base.clone(),
                delta.sequence_value_references.clone(),
            )
        };
        if operations.is_empty() && sequence_value_references.is_empty() {
            drop(commit);
            return self
                .finish_transaction_context(txn_id, true, chain)
                .map_err(ExecuteError::Txn);
        }
        for (ordinal, operation) in operations.iter().enumerate() {
            let stored = match operation {
                TransactionOperation::Catalog(staged) => Some(staged.ordinal),
                TransactionOperation::TableReset(reset) => Some(reset.ordinal),
                TransactionOperation::Row(_) => None,
            };
            if stored.is_some_and(|stored| usize::try_from(stored).ok() != Some(ordinal)) {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "transaction operation lost its global statement ordinal".to_string(),
                )));
            }
        }
        let catalog_commands = operations
            .iter()
            .filter_map(|operation| match operation {
                TransactionOperation::Catalog(staged) => Some(staged.as_ref().clone()),
                TransactionOperation::Row(_) | TransactionOperation::TableReset(_) => None,
            })
            .collect::<Vec<_>>();
        let all_deltas = transaction_row_deltas(&operations);
        let (deltas, table_resets) = final_transaction_operations(&operations);
        let final_staged_rows = final_transaction_row_operations(&operations);
        let has_sequence_resets = operations.iter().any(|operation| {
            matches!(
                operation,
                TransactionOperation::TableReset(reset)
                    if reset.sequence_reset_identity.is_some()
            )
        });
        if !catalog_commands.is_empty() || has_sequence_resets {
            let base = catalog_base.ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(
                    "transactional catalog operation envelope lost its base generation".to_string(),
                ))
            })?;
            if !Self::catalogs_same_ignoring_referenced_sequence_values(
                self.read_state.latest_catalog().as_ref(),
                base.as_ref(),
                &sequence_value_references,
            ) {
                return Err(ExecuteError::Serialization(
                    "catalog contents changed after transactional DDL staging".to_string(),
                ));
            }
            debug_assert!(catalog_commands.iter().all(|operation| {
                Self::transaction_catalog_command_is_supported(&operation.command)
            }));
        }
        let transaction_catalog = snapshot.transaction_catalog();
        for staged in &final_staged_rows {
            let delta = &staged.delta;
            for (name, expected) in &delta.catalog_dependencies {
                if transaction_catalog
                    .relational_catalog
                    .get(name)
                    .is_none_or(|observed| {
                        !same_transaction_row_catalog_dependency(
                            expected,
                            observed,
                            &staged.sequence_input_oids,
                            &transaction_catalog,
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
        }
        let published_catalog = self.read_state.latest_catalog();
        for delta in &deltas {
            if let Some((name, expected)) =
                delta.catalog_dependencies.iter().find(|(name, expected)| {
                    snapshot.catalog.relational_catalog.get(name.as_str()) == Some(*expected)
                        && published_catalog.relational_catalog.get(name.as_str())
                            != Some(*expected)
                })
            {
                return Err(ExecuteError::Serialization(format!(
                    "catalog dependency \"{name}\" changed stable identity or schema before transaction commit (expected OID {})",
                    expected.oid
                )));
            }
        }
        let reset_catalog = if catalog_commands.is_empty() && !has_sequence_resets {
            published_catalog.as_ref()
        } else {
            transaction_catalog.as_ref()
        };
        self.validate_transaction_table_resets(reset_catalog, &table_resets, &commit.ledger)?;
        let staged_tables = deltas
            .iter()
            .map(|delta| match &delta.mutation {
                PreparedMutation::Insert { table, .. }
                | PreparedMutation::Update { table, .. }
                | PreparedMutation::Delete { table, .. } => table,
            })
            .chain(table_resets.iter().map(|reset| &reset.table))
            .collect::<BTreeSet<_>>();
        if let Some(table) = staged_tables.iter().find(|table| {
            snapshot
                .catalog
                .relational_catalog
                .contains_key(table.as_str())
                && !snapshot
                    .chunk_authoritative_tables
                    .contains_key(table.as_str())
                && self.table_chunk_authoritative(table).is_some()
        }) {
            return Err(ExecuteError::Serialization(format!(
                "relation \"{table}\" became cold-chunk authoritative after transaction staging"
            )));
        }
        if let Some(table) = staged_tables.iter().find(|table| {
            snapshot
                .catalog
                .relational_catalog
                .contains_key(table.as_str())
                && !self.table_has_live_dml_generation(table)
        }) {
            return Err(ExecuteError::Serialization(format!(
                "relation \"{table}\" lost its globally authoritative GPU generation after transaction staging"
            )));
        }

        // A later statement may update/delete a row inserted earlier in this transaction. Its
        // provisional row key is not a real conflict point: nobody outside this transaction can
        // observe that entity, and COMMIT assigns it a fresh globally-claimed identity.
        let provisional_inserts = Self::transaction_insert_identities(&all_deltas)?;
        self.validate_transaction_device_row_conflicts(&snapshot, &deltas, &provisional_inserts)?;

        // The canonical commit mutex excludes every other row-id claimant. Use its current value as
        // the tentative record base, validate the complete unique/FK outcome, and let canonical
        // apply advance to the encoded high-water. A rejected pre-WAL transaction therefore cannot
        // leak process-wide row identities.
        let insert_count = provisional_inserts.len() as u64;
        let final_base = self.read_state.mvcc.current_row_id();
        let allocator_high_water = final_base.checked_add(insert_count).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(
                "transaction row identity space exhausted".to_string(),
            ))
        })?;
        let mut record = Self::resolved_transaction_record(
            &all_deltas,
            &deltas,
            &provisional_inserts,
            final_base,
            allocator_high_water,
            &sequence_value_references,
        )?;
        Self::bind_transaction_record_catalog_envelope(
            &mut record,
            &operations,
            &catalog_commands,
            &table_resets,
            &transaction_catalog,
        )?;
        Self::bind_sequence_reference_final_dispositions(&mut record, &transaction_catalog)?;
        let cold_index_candidate_tables = record
            .index_lifecycle_operations
            .iter()
            .flat_map(|operation| &operation.targets)
            .filter_map(|target| target.owner_name.clone())
            .collect::<BTreeSet<_>>();
        if !record.catalog_commands.is_empty() || !record.sequence_reset_operations.is_empty() {
            self.validate_transaction_catalog_before_wal(
                record.catalog_epoch,
                &record.catalog_commands,
                TransactionCatalogEnvelopeSlices {
                    view_operations: &record.view_operations,
                    view_lifecycle_operations: &record.view_lifecycle_operations,
                    index_lifecycle_operations: &record.index_lifecycle_operations,
                    sequence_lifecycle_operations: &record.sequence_lifecycle_operations,
                    sequence_reset_operations: &record.sequence_reset_operations,
                    operation_order: &record.operation_order,
                    catalog_output: record.catalog_output.as_ref(),
                    sequence_input_oids: &record.sequence_input_oids,
                    sequence_value_references: &record.sequence_value_references,
                },
                &transaction_catalog,
            )?;
        }
        self.validate_transaction_surviving_created_unique_indexes_device(
            &snapshot,
            &transaction_catalog,
            &record.catalog_commands,
            &record.index_lifecycle_operations,
        )?;
        let reset_tables = table_resets
            .iter()
            .map(|reset| reset.table.clone())
            .collect::<BTreeSet<_>>();
        self.validate_transaction_device_unique_conflicts(
            &snapshot,
            &deltas,
            &record,
            &reset_tables,
        )?;
        self.validate_transaction_device_foreign_key_conflicts(
            &snapshot,
            &deltas,
            &record,
            &commit.ledger,
        )?;
        let payload: Arc<[u8]> =
            Arc::from(try_encode_binary_transaction(&record).ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(
                    "resolved transaction WAL record exceeds binary framing limits".to_string(),
                ))
            })?);
        // Close every persistent GPU allocation before WAL. The exact final record—not staged
        // intermediate versions—defines the one canonical publication batch. Existing private
        // allocation charges become post-durable credit; reserve only any remaining difference.
        // Keep enrolled named-index coverage stable from restoration through canonical apply so a
        // racing invalidation cannot turn an admitted durable commit into a post-WAL failure.
        let named_index_tables = record
            .mutations
            .iter()
            .map(|mutation| match mutation {
                BinaryTransactionMutation::Insert { table, .. }
                | BinaryTransactionMutation::Update { table, .. }
                | BinaryTransactionMutation::Delete { table, .. } => table.clone(),
            })
            .chain(record.catalog_commands.iter().filter_map(
                |operation| match &operation.command {
                    Command::CreateTable(create) => Some(create.table.clone()),
                    _ => None,
                },
            ))
            .chain(
                record
                    .index_lifecycle_operations
                    .iter()
                    .flat_map(|operation| &operation.targets)
                    .filter_map(|target| target.owner_name.clone()),
            )
            .chain(record.table_resets.iter().map(|reset| reset.table.clone()))
            .collect::<BTreeSet<_>>();
        let mut named_index_publication = self
            .read_state
            .residency
            .begin_transaction_named_index_publication(named_index_tables);
        self.reserve_transaction_canonical_publication(&snapshot, &transaction_catalog, &record)?;
        let affected_rows =
            Self::canonical_affected_rows(&payload).map_err(ExecuteError::Engine)?;
        let request_digest = request_digest_override
            .unwrap_or_else(|| gpu_db_wal::canonical_request_digest(&payload));
        let isolation = match snapshot.characteristics.isolation {
            TransactionIsolation::ReadUncommitted | TransactionIsolation::ReadCommitted => {
                gpu_db_wal::CanonicalIsolation::ReadCommitted
            }
            TransactionIsolation::RepeatableRead => gpu_db_wal::CanonicalIsolation::RepeatableRead,
            TransactionIsolation::Serializable => {
                return Err(ExecuteError::Unsupported(
                    "SERIALIZABLE transactions are not implemented".to_string(),
                ));
            }
        };
        // Register the chained successor before the first durable effect. Allocator exhaustion is
        // therefore a clean pre-WAL error that leaves the parent transaction and its lifetime
        // snapshot active. Every later failure path cancels this provisional successor.
        let successor_id = if chain {
            Some(
                self.begin_unclaimed_transaction(&mut commit)
                    .map_err(ExecuteError::Txn)?,
            )
        } else {
            None
        };

        let wal_len_before = commit.wal.len();
        let token = match commit.repl.propose(Arc::clone(&payload)) {
            Ok(token) => token,
            Err(err) => {
                Self::cancel_chained_successor(&mut commit, successor_id);
                return Err(ExecuteError::Engine(err));
            }
        };
        let record = match Self::canonical_wal_record_with_isolation_and_request_digest(
            &commit,
            txn_id,
            token.index,
            0,
            &payload,
            isolation,
            request_digest,
        ) {
            Ok(record) => record,
            Err(err) => {
                commit.repl.rollback_unapplied_from(token.index);
                Self::cancel_chained_successor(&mut commit, successor_id);
                return Err(ExecuteError::Engine(err));
            }
        };
        commit.wal.append(record);
        if let Err(err) = commit.wal.flush_all() {
            commit.repl.rollback_unapplied_from(token.index);
            commit.wal.truncate(wal_len_before);
            Self::cancel_chained_successor(&mut commit, successor_id);
            return Err(ExecuteError::Engine(err));
        }
        if let Err(error) = commit.repl.wait_committed(token, Duration::from_millis(0)) {
            Self::cancel_chained_successor(&mut commit, successor_id);
            self.wedge_commit_path();
            return Err(ExecuteError::Indeterminate(format!(
                "explicit transaction {txn_id} flushed WAL but commit confirmation failed: {error}; restart recovery must resolve it"
            )));
        }
        if let Err(error) = commit.record_transaction_status_digest_outcome(
            txn_id,
            request_digest,
            token.index,
            affected_rows,
        ) {
            Self::cancel_chained_successor(&mut commit, successor_id);
            self.wedge_commit_path();
            return Err(ExecuteError::Indeterminate(format!(
                "explicit transaction {txn_id} became durable but terminal status installation failed: {error}; restart recovery must resolve it"
            )));
        }
        if request_digest_override.is_some() {
            self.release_pending_transaction_claim(txn_id, request_digest);
        }
        commit.record_commit_timestamp(txn_id, timestamp_micros);
        // The durable resolved record is now the sole source of truth. Retire transaction-private
        // COW allocations before canonical apply constructs/publishes the global generation; this
        // avoids double-accounting the same logical rows against the GPU residency budget.
        let commit_gpu_credit = self.retire_transaction_private_generation_post_durable(&snapshot);
        #[cfg(test)]
        let post_durable_hook = self
            .transaction_post_durable_hook
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        #[cfg(test)]
        if let Some(hook) = post_durable_hook {
            hook();
        }
        // From canonical apply's first allocation through acknowledgement, an external
        // pressure/retirement request must outlive the final index publication. Owner-local
        // replacement purges use the explicit during-transaction bypass and are not replayed.
        named_index_publication.enter_final_publication();
        let publication_owner = TransactionNamedIndexPublicationOwnerGuard::enter();
        #[cfg(test)]
        let apply_result = self.with_transaction_commit_gpu_credit(&commit_gpu_credit, || {
            if self
                .fail_next_transaction_post_durable_apply
                .swap(false, AtomicOrdering::AcqRel)
            {
                Err(EngineError::ApplyFailed(
                    "injected post-durable explicit-transaction apply failure".to_string(),
                ))
            } else {
                self.apply_and_publish_committed(&mut commit, txn_id, token.index)
            }
        });
        #[cfg(not(test))]
        let apply_result = self.with_transaction_commit_gpu_credit(&commit_gpu_credit, || {
            self.apply_and_publish_committed(&mut commit, txn_id, token.index)
        });
        drop(publication_owner);
        self.release_transaction_commit_gpu_credit(&commit_gpu_credit);
        if let Err(err) = apply_result {
            Self::cancel_chained_successor(&mut commit, successor_id);
            self.wedge_commit_path();
            return Err(ExecuteError::Indeterminate(format!(
                "explicit transaction {txn_id} is durable but could not be fully installed: {err}; engine restart recovery required"
            )));
        }
        #[cfg(test)]
        self.read_state
            .residency
            .run_transaction_named_index_post_apply_hook();
        named_index_publication.complete();

        commit
            .txn_manager
            .commit(txn_id)
            .expect("active transaction remained active under the commit lock");
        let successor = successor_id.map(|next_id| {
            (
                next_id,
                self.capture_transaction_snapshot(self.committed_seq(), snapshot.characteristics),
            )
        });
        let mut active = self
            .active_snapshots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let removed = active.deregister_transaction(txn_id);
        debug_assert!(removed.is_some(), "committed transaction lost its snapshot");
        let successor_id = successor.as_ref().map(|(next_id, _)| *next_id);
        if let Some((next_id, next_snapshot)) = successor {
            active.register_transaction(next_id, next_snapshot);
        }
        drop(active);
        self.gc_transaction_created_by_regions();
        self.metrics.inc_commit();
        drop(commit);

        // Cold candidate directories are GPU structures, but spilled payload staging may issue
        // positional NVMe reads. Lifecycle publication therefore retires the obsolete directories
        // while holding the sole commit/publication lock, then rebuilds the final catalog's unique
        // key candidates only after releasing it. COMMIT does not return until this best-effort
        // priming pass finishes; a concurrent DML that reaches an unprimed spilled generation
        // during this narrow interval fails closed instead of consulting any host index.
        let cold_chunks = self.read_streaming_cold_chunks();
        for table_name in cold_index_candidate_tables {
            if let Some(entry) = cold_chunks.get(&table_name) {
                self.prime_chunk_key_candidates(&table_name, entry);
            }
        }
        Ok(successor_id)
    }

    fn retire_transaction_private_generation_post_durable(
        &self,
        snapshot: &TransactionSnapshot,
    ) -> BTreeMap<u16, u64> {
        let (private_shards, private_cold_chunks, charges, commit_charges) = {
            let mut delta = snapshot
                .delta
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            assert!(
                Arc::ptr_eq(&delta.resident_shards, &delta.resident_shards_authority)
                    && Arc::ptr_eq(
                        &delta.streaming_cold_chunks,
                        &delta.streaming_cold_chunks_authority
                    ),
                "pre-WAL validation must seal the private GPU generation before retirement"
            );
            let private_shards = Arc::clone(&delta.resident_shards);
            let private_cold_chunks = Arc::clone(&delta.streaming_cold_chunks);
            delta.publish_resident_shards(Arc::clone(&snapshot.resident_shards));
            delta.publish_streaming_cold_chunks(Arc::clone(&snapshot.base_streaming_cold_chunks));
            let charges = std::mem::take(&mut delta.private_gpu_bytes_by_gpu);
            let commit_charges = std::mem::take(&mut delta.commit_gpu_bytes_by_gpu);
            (private_shards, private_cold_chunks, charges, commit_charges)
        };
        let mut credit = charges;
        for (gpu_id, bytes) in commit_charges {
            let slot = credit.entry(gpu_id).or_default();
            *slot = slot.saturating_add(bytes);
        }
        // Keep the exact account charge alive as a publication credit while the private Arcs are
        // dropped. Concurrent allocators continue to see it; only this apply thread subtracts it
        // from its own budget observations until global payload/index allocations replace it.
        drop(private_shards);
        drop(private_cold_chunks);
        credit
    }

    fn validate_transaction_device_row_conflicts(
        &self,
        snapshot: &TransactionSnapshot,
        deltas: &[WriteDelta],
        provisional_inserts: &BTreeSet<(String, u64)>,
    ) -> Result<(), ExecuteError> {
        debug_assert!(
            self.current_transaction_read_snapshot().is_none(),
            "COMMIT conflict validation must bind the current device generation"
        );
        let current_boundary = self.committed_seq();
        let mut checked = BTreeSet::new();
        for delta in deltas {
            let read_snapshot = delta.read_snapshot;
            let (table_name, targets): (&str, Vec<(&str, &[SqlValue])>) = match &delta.mutation {
                PreparedMutation::Insert { .. } => continue,
                PreparedMutation::Update {
                    table,
                    installs,
                    updated_old_rows,
                    ..
                } => (
                    table,
                    installs
                        .iter()
                        .zip(updated_old_rows)
                        .map(|((_, key, _), row)| (key.as_str(), row.as_slice()))
                        .collect(),
                ),
                PreparedMutation::Delete {
                    table,
                    deleted_rows,
                    ..
                } => (
                    table,
                    delta
                        .write_set
                        .rows
                        .iter()
                        .zip(deleted_rows)
                        .map(|(key, row)| (key.row_key.as_str(), row.as_slice()))
                        .collect(),
                ),
            };
            let catalog = snapshot.transaction_catalog();
            let table = catalog
                .relational_catalog
                .get(table_name)
                .expect("staged transaction table remains in its retained catalog");
            let prefix = relational_key_prefix(table_name);
            for (row_key, expected_row) in targets {
                let row_id = crate::engine_residency::parse_relational_row_id(row_key, &prefix)
                    .ok_or_else(|| {
                        ExecuteError::Engine(EngineError::ApplyFailed(
                            "transaction conflict target lost stable entity identity".to_string(),
                        ))
                    })?;
                let identity = (table_name.to_string(), row_id);
                if provisional_inserts.contains(&identity) || !checked.insert(identity) {
                    continue;
                }
                if snapshot.chunk_authoritative_tables.contains_key(table_name) {
                    let Some((observed, created_by, _, _)) =
                        self.class_row_by_entity_identity(table, row_id, current_boundary)
                    else {
                        return Err(ExecuteError::Serialization(format!(
                            "device write-write conflict on entity {row_id} in relation \"{table_name}\" after snapshot {}",
                            read_snapshot
                        )));
                    };
                    if created_by > read_snapshot || observed.as_slice() != expected_row {
                        return Err(ExecuteError::Serialization(format!(
                            "device write-write conflict on entity {row_id} in relation \"{table_name}\" after snapshot {}",
                            read_snapshot
                        )));
                    }
                    continue;
                }
                let filters = expected_row
                    .iter()
                    .enumerate()
                    .map(|(idx, value)| (idx, SelectFilterOp::Eq, value.clone()))
                    .collect::<Vec<_>>();
                let probe = self.dml_device_probe_key(table, &filters);
                let indexed = probe.and_then(|(key_id, needle)| {
                    self.locate_resident_pk_via_shard_index_detailed(table, key_id, needle)
                });
                let hits = match indexed {
                    Some(hits) if !hits.is_empty() => hits,
                    _ if crate::engine_prepared_transaction::prepared_index_route_required() => {
                        // The engine-owned proof promised an exact named-index route. Version-chain
                        // overflow or any physical decline is a retryable conflict, never permission
                        // to introduce an undeclared O(rows) commit-time scan.
                        return Err(ExecuteError::Serialization(format!(
                            "prepared device write-conflict index verdict unavailable for relation \"{table_name}\""
                        )));
                    }
                    _ => {
                        // Version churn can make a unique index decline because old and new
                        // physical slots share a key. Scan the same probe key structurally when
                        // possible; it may return multiple versions, which the identity, creation
                        // stamp, and full-row materialization checks below disambiguate exactly.
                        let key_cols = if let Some((key_id, _)) = probe {
                            let Some(value) = expected_row.get(key_id) else {
                                return Err(ExecuteError::Serialization(format!(
                                    "device write-conflict predicate unavailable for relation \"{table_name}\""
                                )));
                            };
                            vec![(key_id, value.clone())]
                        } else {
                            expected_row
                                .iter()
                                .enumerate()
                                .map(|(idx, value)| (idx, value.clone()))
                                .collect::<Vec<_>>()
                        };
                        let Some(predicates) =
                            crate::engine_dml_prepare::device_structural_tuple_predicates(
                                table, &key_cols,
                            )
                        else {
                            return Err(ExecuteError::Serialization(format!(
                                "device write-conflict predicate unavailable for relation \"{table_name}\""
                            )));
                        };
                        let Some(hits) = self
                            .locate_resident_conjunct_slots_detailed(table, &predicates)
                            .or_else(|| self.locate_resident_all_slots_detailed(table))
                        else {
                            return Err(ExecuteError::Serialization(format!(
                                "device write-conflict locate unavailable for relation \"{table_name}\""
                            )));
                        };
                        hits
                    }
                };
                let mut matched = 0usize;
                for hit in hits {
                    if self.hit_entity_id(&hit) != Some(row_id) {
                        continue;
                    }
                    let created_by = match &hit.created_by {
                        None => 0,
                        Some(region) => {
                            let halves = region
                                .read_resident_i32_column(u64::from(hit.slot) * 8, 2)
                                .map_err(|err| {
                                    ExecuteError::Serialization(format!(
                                        "device write-conflict stamp read failed for relation \"{table_name}\": {err}"
                                    ))
                                })?;
                            (halves[0] as u32 as u64) | ((halves[1] as u32 as u64) << 32)
                        }
                    };
                    if created_by > read_snapshot {
                        return Err(ExecuteError::Serialization(format!(
                            "device write-write conflict on entity {row_id} in relation \"{table_name}\" after snapshot {}",
                            read_snapshot
                        )));
                    }
                    match self.materialize_resident_row_via_hit(table, &hit, current_boundary) {
                        Some(Some(observed)) if observed.as_slice() == expected_row => matched += 1,
                        Some(Some(_)) | Some(None) => {}
                        None => {
                            return Err(ExecuteError::Serialization(format!(
                                "device write-conflict row verification failed for relation \"{table_name}\""
                            )))
                        }
                    }
                }
                if matched != 1 {
                    return Err(ExecuteError::Serialization(format!(
                        "device write-write conflict on entity {row_id} in relation \"{table_name}\" after snapshot {}",
                        read_snapshot
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_transaction_device_unique_conflicts(
        &self,
        snapshot: &TransactionSnapshot,
        deltas: &[WriteDelta],
        record: &BinaryTransactionRecord,
        reset_tables: &BTreeSet<String>,
    ) -> Result<(), ExecuteError> {
        // Every UPDATE/DELETE identity retires its old unique-key ownership at this transaction's
        // single publish boundary. Candidate final rows (including INSERTs) must exclude the whole
        // retiring set, not merely an UPDATE's own identity: key release+reuse and key swaps are
        // valid when the transaction's final relation is unique.
        let mut retiring_ids = BTreeMap::<String, BTreeSet<u64>>::new();
        for mutation in &record.mutations {
            match mutation {
                BinaryTransactionMutation::Update { table, row_id, .. }
                | BinaryTransactionMutation::Delete { table, row_id, .. } => {
                    retiring_ids
                        .entry(table.clone())
                        .or_default()
                        .insert(*row_id);
                }
                BinaryTransactionMutation::Insert { .. } => {}
            }
        }
        // First-committer-wins includes released keys, not only current duplicates. Scan every old
        // and new exact key against physical device version history before the current-row check
        // below; an insert+delete or key-away after BEGIN must still serialize this transaction.
        for delta in deltas {
            let (table_name, rows): (&str, Vec<&[SqlValue]>) = match &delta.mutation {
                PreparedMutation::Insert {
                    table,
                    inserted_rows,
                    ..
                } => (
                    table,
                    inserted_rows
                        .iter()
                        .map(|(_, row)| row.as_slice())
                        .collect(),
                ),
                PreparedMutation::Update {
                    table,
                    installs,
                    updated_old_rows,
                    ..
                } => (
                    table,
                    updated_old_rows
                        .iter()
                        .map(Vec::as_slice)
                        .chain(installs.iter().map(|(_, _, row)| row.as_slice()))
                        .collect(),
                ),
                PreparedMutation::Delete {
                    table,
                    deleted_rows,
                    ..
                } => (table, deleted_rows.iter().map(Vec::as_slice).collect()),
            };
            let catalog = snapshot.transaction_catalog();
            let table = catalog
                .relational_catalog
                .get(table_name)
                .expect("transaction record table remains in retained catalog");
            if reset_tables.contains(table_name) {
                continue;
            }
            if !table.indexes.iter().any(|index| index.unique)
                || !snapshot.catalog.relational_catalog.contains_key(table_name)
            {
                continue;
            }
            let conflict_boundary = self.transaction_table_conflict_boundary(
                snapshot,
                table_name,
                delta.read_snapshot,
            )?;
            match self.device_unique_rows_conflict(table, &rows, conflict_boundary) {
                Some(false) => {}
                Some(true) => {
                    return Err(ExecuteError::Serialization(format!(
                        "device unique conflict: write history changed in relation \"{table_name}\" after snapshot {}",
                        conflict_boundary
                    )))
                }
                None => {
                    return Err(ExecuteError::Serialization(format!(
                        "device unique-history verdict unavailable for relation \"{table_name}\""
                    )))
                }
            }
        }
        let visibility = StorageVisibility {
            read_txn_id: self.committed_seq(),
        };
        let catalog = snapshot.transaction_catalog();
        for mutation in &record.mutations {
            let (table_name, row_id, encoded) = match mutation {
                BinaryTransactionMutation::Insert {
                    table,
                    row_id,
                    row_encoded,
                } => (table, *row_id, row_encoded),
                BinaryTransactionMutation::Update {
                    table,
                    row_id,
                    new_row_encoded,
                    ..
                } => (table, *row_id, new_row_encoded),
                BinaryTransactionMutation::Delete { .. } => continue,
            };
            let table = catalog
                .relational_catalog
                .get(table_name)
                .expect("transaction record table remains in retained catalog");
            if reset_tables.contains(table_name) {
                continue;
            }
            if !snapshot.catalog.relational_catalog.contains_key(table_name) {
                continue;
            }
            if !table.indexes.iter().any(|index| index.unique) {
                continue;
            }
            let row = decode_relational_row(encoded, &table.columns).map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "transaction unique image decode failed for relation \"{table_name}\": {err}"
                )))
            })?;
            if self.table_chunk_authoritative(table_name).is_some() {
                let exclusion = retiring_ids.get(table_name).map(|ids| {
                    let mut coordinates = BTreeSet::new();
                    let mut epoch = None;
                    for retiring_id in ids {
                        let (_, _, coordinate, observed_epoch) = self
                            .class_row_by_entity_identity(
                                table,
                                *retiring_id,
                                visibility.read_txn_id,
                            )
                            .ok_or_else(|| {
                                ExecuteError::Serialization(format!(
                                    "device retiring-identity verdict unavailable for entity {retiring_id} in relation \"{table_name}\""
                                ))
                            })?;
                        if epoch.is_some_and(|current| current != observed_epoch) {
                            return Err(ExecuteError::Serialization(format!(
                                "device retiring identities crossed class generations in relation \"{table_name}\""
                            )));
                        }
                        epoch = Some(observed_epoch);
                        coordinates.insert(coordinate);
                    }
                    Ok((coordinates, epoch.unwrap_or(0)))
                }).transpose()?;
                let exclusion_ref = exclusion
                    .as_ref()
                    .map(|(coordinates, epoch)| (coordinates, *epoch));
                match self.validate_class_insert_uniqueness(
                    table,
                    std::slice::from_ref(&row),
                    visibility.read_txn_id,
                    exclusion_ref,
                ) {
                    Some(Ok(())) => {}
                    Some(Err(_)) => {
                        return Err(ExecuteError::Serialization(format!(
                            "device unique conflict on entity {row_id} in relation \"{table_name}\""
                        )))
                    }
                    None => {
                        return Err(ExecuteError::Serialization(format!(
                        "device unique-conflict verdict unavailable for relation \"{table_name}\""
                    )))
                    }
                }
                continue;
            }

            let excluded = retiring_ids.get(table_name).map(|ids| {
                ids.iter()
                    .map(|retiring_id| relational_row_key(table_name, *retiring_id))
                    .collect::<BTreeSet<_>>()
            });
            for (ordinal, index) in table
                .indexes
                .iter()
                .enumerate()
                .filter(|(_, index)| index.unique)
            {
                let positions = crate::engine_residency::index_key_column_positions(table, index)
                    .ok_or_else(|| {
                        ExecuteError::Serialization(format!(
                            "device unique-conflict key binding unavailable for relation \"{table_name}\""
                        ))
                    })?;
                let conflict = if crate::engine_residency::index_uses_fingerprint(table, index) {
                    let key_id = crate::engine_residency::index_probe_key_id(table, index, ordinal)
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                            "device unique-conflict index unavailable for relation \"{table_name}\""
                        ))
                        })?;
                    let key_columns = positions
                        .iter()
                        .map(|position| (*position, row[*position].clone()))
                        .collect::<Vec<_>>();
                    match crate::engine_residency::compound_index_row_fingerprint(
                        table, index, &row,
                    ) {
                        Some(fingerprint) => self.device_visible_row_with_tuple(
                            table,
                            visibility,
                            key_id,
                            Some(fingerprint),
                            &key_columns,
                            excluded.as_ref(),
                        ),
                        None if positions.len() == 1 => self.device_visible_row_with_value(
                            table,
                            visibility,
                            positions[0],
                            &row[positions[0]],
                            excluded.as_ref(),
                        ),
                        None => self.device_visible_row_with_tuple(
                            table,
                            visibility,
                            key_id,
                            None,
                            &key_columns,
                            excluded.as_ref(),
                        ),
                    }
                } else {
                    self.device_visible_row_with_value(
                        table,
                        visibility,
                        positions[0],
                        &row[positions[0]],
                        excluded.as_ref(),
                    )
                };
                match conflict {
                    Some(false) => {}
                    Some(true) => {
                        return Err(ExecuteError::Serialization(format!(
                            "device unique conflict on entity {row_id} in relation \"{table_name}\""
                        )))
                    }
                    None => {
                        return Err(ExecuteError::Serialization(format!(
                        "device unique-conflict verdict unavailable for relation \"{table_name}\""
                    )))
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_transaction_device_foreign_key_conflicts(
        &self,
        snapshot: &Arc<TransactionSnapshot>,
        deltas: &[WriteDelta],
        record: &BinaryTransactionRecord,
        ledger: &RecentCommitsLedger,
    ) -> Result<(), ExecuteError> {
        for delta in deltas {
            for table in &delta.foreign_key_dependencies {
                let conflict_boundary =
                    self.transaction_table_conflict_boundary(snapshot, table, delta.read_snapshot)?;
                if ledger.table_changed_after(table, conflict_boundary) {
                    return Err(ExecuteError::Serialization(format!(
                        "foreign-key dependency relation \"{table}\" changed after transaction conflict boundary {conflict_boundary}"
                    )));
                }
            }
        }
        type IdentityRows = BTreeMap<String, Vec<(u64, Vec<SqlValue>)>>;
        let catalog = snapshot.transaction_catalog();
        let mut final_rows = IdentityRows::new();
        let mut removed_rows = IdentityRows::new();
        let mut mutated_ids = BTreeMap::<String, BTreeSet<u64>>::new();
        for mutation in &record.mutations {
            let (table_name, row_id) = match mutation {
                BinaryTransactionMutation::Insert { table, row_id, .. }
                | BinaryTransactionMutation::Update { table, row_id, .. }
                | BinaryTransactionMutation::Delete { table, row_id, .. } => (table, *row_id),
            };
            let table = catalog
                .relational_catalog
                .get(table_name)
                .expect("transaction FK table remains in retained catalog");
            mutated_ids
                .entry(table_name.clone())
                .or_default()
                .insert(row_id);
            match mutation {
                BinaryTransactionMutation::Insert { row_encoded, .. } => {
                    let row = decode_relational_row(row_encoded, &table.columns).map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(format!(
                            "transaction FK insert image decode failed for relation \"{table_name}\": {err}"
                        )))
                    })?;
                    final_rows
                        .entry(table_name.clone())
                        .or_default()
                        .push((row_id, row));
                }
                BinaryTransactionMutation::Update {
                    old_row_encoded,
                    new_row_encoded,
                    ..
                } => {
                    let old = decode_relational_row(old_row_encoded, &table.columns).map_err(
                        |err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "transaction FK old image decode failed for relation \"{table_name}\": {err}"
                            )))
                        },
                    )?;
                    let new = decode_relational_row(new_row_encoded, &table.columns).map_err(
                        |err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "transaction FK new image decode failed for relation \"{table_name}\": {err}"
                            )))
                        },
                    )?;
                    removed_rows
                        .entry(table_name.clone())
                        .or_default()
                        .push((row_id, old));
                    final_rows
                        .entry(table_name.clone())
                        .or_default()
                        .push((row_id, new));
                }
                BinaryTransactionMutation::Delete {
                    old_row_encoded, ..
                } => {
                    let old = decode_relational_row(old_row_encoded, &table.columns).map_err(
                        |err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                                "transaction FK delete image decode failed for relation \"{table_name}\": {err}"
                            )))
                        },
                    )?;
                    removed_rows
                        .entry(table_name.clone())
                        .or_default()
                        .push((row_id, old));
                }
            }
        }
        let visibility = StorageVisibility {
            read_txn_id: self.committed_seq(),
        };

        // Outbound stamps: untouched current providers are checked against the current device
        // generation. Providers written by this transaction must exist in its final private
        // device generation; the final-row stamp prevents an old BEGIN version from satisfying it.
        for (child_name, rows) in &final_rows {
            let child = catalog
                .relational_catalog
                .get(child_name)
                .expect("transaction child remains cataloged");
            for foreign_key in &child.foreign_keys {
                let Some(parent) = catalog
                    .relational_catalog
                    .get(&foreign_key.referenced_table)
                else {
                    continue;
                };
                let child_idx =
                    relational_column_index(child, &foreign_key.column).map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                let parent_idx = relational_column_index(parent, &foreign_key.referenced_column)
                    .map_err(|err| {
                        ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                    })?;
                let parent_exclusions = mutated_ids.get(&parent.name).map(|ids| {
                    ids.iter()
                        .map(|row_id| relational_row_key(&parent.name, *row_id))
                        .collect::<BTreeSet<_>>()
                });
                for (_, row) in rows {
                    let value = &row[child_idx];
                    if matches!(value, SqlValue::Null) {
                        continue;
                    }
                    let current = self
                        .device_visible_row_with_value(
                            parent,
                            visibility,
                            parent_idx,
                            value,
                            parent_exclusions.as_ref(),
                        )
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                                "device foreign-key provider verdict unavailable for relation \"{}\"",
                                parent.name
                            ))
                        })?;
                    if current {
                        continue;
                    }
                    let final_provider_stamp =
                        final_rows.get(&parent.name).is_some_and(|providers| {
                            providers
                                .iter()
                                .any(|(_, provider)| provider[parent_idx] == *value)
                        });
                    let private_provider = if final_provider_stamp {
                        let _scope = self.enter_transaction_read(Arc::clone(snapshot));
                        self.device_visible_row_with_value(
                            parent,
                            StorageVisibility {
                                read_txn_id: snapshot.boundary,
                            },
                            parent_idx,
                            value,
                            None,
                        )
                        .ok_or_else(|| {
                            ExecuteError::Serialization(format!(
                                "private device foreign-key provider verdict unavailable for relation \"{}\"",
                                parent.name
                            ))
                        })?
                    } else {
                        false
                    };
                    if !private_provider {
                        return Err(ExecuteError::Serialization(format!(
                            "device foreign-key conflict on constraint \"{}\" in relation \"{}\"",
                            foreign_key.name, child.name
                        )));
                    }
                }
            }
        }

        // Inbound stamps: final private child images are explicit conflicts; current untouched
        // children are checked on-device with transaction-removed identities excluded.
        for (parent_name, rows) in &removed_rows {
            let parent = catalog
                .relational_catalog
                .get(parent_name)
                .expect("transaction parent remains cataloged");
            for child in catalog.relational_catalog.values() {
                for foreign_key in child
                    .foreign_keys
                    .iter()
                    .filter(|foreign_key| foreign_key.referenced_table == *parent_name)
                {
                    let parent_idx =
                        relational_column_index(parent, &foreign_key.referenced_column).map_err(
                            |err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())),
                        )?;
                    let child_idx =
                        relational_column_index(child, &foreign_key.column).map_err(|err| {
                            ExecuteError::Engine(EngineError::ApplyFailed(err.to_string()))
                        })?;
                    let child_exclusions = mutated_ids.get(&child.name).map(|ids| {
                        ids.iter()
                            .map(|row_id| relational_row_key(&child.name, *row_id))
                            .collect::<BTreeSet<_>>()
                    });
                    let final_children = final_rows.get(&child.name);
                    let final_providers = final_rows.get(parent_name);
                    let mut checked = BTreeSet::<SqlValue>::new();
                    for (_, old_parent) in rows {
                        let value = old_parent[parent_idx].clone();
                        if matches!(value, SqlValue::Null) || !checked.insert(value.clone()) {
                            continue;
                        }
                        if final_providers.is_some_and(|providers| {
                            providers
                                .iter()
                                .any(|(_, provider)| provider[parent_idx] == value)
                        }) {
                            continue;
                        }
                        if final_children.is_some_and(|children| {
                            children
                                .iter()
                                .any(|(_, child_row)| child_row[child_idx] == value)
                        }) {
                            return Err(ExecuteError::Serialization(format!(
                                "device foreign-key conflict on constraint \"{}\" in relation \"{}\"",
                                foreign_key.name, child.name
                            )));
                        }
                        let surviving_child = self
                            .device_visible_row_with_value(
                                child,
                                visibility,
                                child_idx,
                                &value,
                                child_exclusions.as_ref(),
                            )
                            .ok_or_else(|| {
                                ExecuteError::Serialization(format!(
                                    "device foreign-key child verdict unavailable for relation \"{}\"",
                                    child.name
                                ))
                            })?;
                        if surviving_child {
                            return Err(ExecuteError::Serialization(format!(
                                "device foreign-key conflict on constraint \"{}\" in relation \"{}\"",
                                foreign_key.name, child.name
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Keep BEGIN-generation preparation and its cheap conflict verdict available to the CPU
    /// parity oracle, but require an actual retained device generation before publishing a private
    /// delta. The CPU store is not a transaction-execution fallback for a conflict-free statement.
    pub(crate) fn validate_transaction_delta_residency(
        &self,
        table: &RelationalTable,
        existed_at_transaction_base: bool,
    ) -> Result<(), ExecuteError> {
        // A transaction-private CREATE owns an empty table generation before its first INSERT.
        // There cannot be a published shard for that relation yet; the private append below builds
        // its first device shard, and canonical apply/replay installs the final committed shard.
        if !existed_at_transaction_base {
            return Ok(());
        }
        if self
            .current_transaction_read_snapshot()
            .is_some_and(|snapshot| snapshot.table_has_typed_empty_root(&table.name))
        {
            return Ok(());
        }
        if self.table_chunk_authoritative(&table.name).is_some() {
            if self.read_streaming_cold_chunks().contains_key(&table.name) {
                return Ok(());
            }
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no retained cold-chunk generation for transaction staging",
                table.name
            ))));
        }
        let shards = self.read_residency_shards();
        if shards
            .get(&table.name)
            .is_none_or(|table_shards| table_shards.is_empty())
        {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" has no retained GPU shard generation for transaction staging",
                table.name
            ))));
        }
        Ok(())
    }

    fn apply_transaction_private_delta(
        &self,
        table: &RelationalTable,
        delta: &WriteDelta,
        boundary: Index,
        shards: &mut BTreeMap<String, Vec<RelationalResidentShard>>,
        cold_chunks: &mut BTreeMap<String, Arc<crate::engine_streaming_exec::ColdTableChunks>>,
        gpu_reservation: &mut TransactionGpuReservation<'_>,
    ) -> Result<(), ExecuteError> {
        if self.table_chunk_authoritative(&table.name).is_some() {
            let prefix = relational_key_prefix(&table.name);
            let entry = cold_chunks.get(&table.name).cloned().ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" lost its retained cold-chunk generation",
                    table.name
                )))
            })?;
            let parse_ids = |keys: Vec<&str>| -> Result<Vec<u64>, ExecuteError> {
                keys.into_iter()
                    .map(|key| {
                        crate::engine_residency::parse_relational_row_id(key, &prefix).ok_or_else(
                            || {
                                ExecuteError::Engine(EngineError::ApplyFailed(
                                    "transaction cold mutation lost stable entity identity"
                                        .to_string(),
                                ))
                            },
                        )
                    })
                    .collect()
            };
            let next = match &delta.mutation {
                PreparedMutation::Insert { inserted_rows, .. } => {
                    let rows = inserted_rows
                        .iter()
                        .map(|(_, row)| row.clone())
                        .collect::<Vec<_>>();
                    let row_ids =
                        parse_ids(inserted_rows.iter().map(|(key, _)| key.as_str()).collect())?;
                    self.append_transaction_cold_tail(table, &entry, &rows, &row_ids, boundary)
                }
                PreparedMutation::Update {
                    installs,
                    class_epoch: Some(epoch),
                    ..
                } => {
                    let coordinates = installs
                        .iter()
                        .map(|(coordinate, _, _)| *coordinate)
                        .collect::<Vec<_>>();
                    let row_ids =
                        parse_ids(installs.iter().map(|(_, key, _)| key.as_str()).collect())?;
                    let rows = installs
                        .iter()
                        .map(|(_, _, row)| row.clone())
                        .collect::<Vec<_>>();
                    self.stamp_transaction_cold_coordinates(&entry, &coordinates, *epoch, boundary)
                        .and_then(|stamped| {
                            self.append_transaction_cold_tail(
                                table, &stamped, &rows, &row_ids, boundary,
                            )
                        })
                }
                PreparedMutation::Delete {
                    tuple_ids,
                    class_epoch: Some(epoch),
                    ..
                } => self.stamp_transaction_cold_coordinates(&entry, tuple_ids, *epoch, boundary),
                PreparedMutation::Update {
                    class_epoch: None, ..
                }
                | PreparedMutation::Delete {
                    class_epoch: None, ..
                } => None,
            }
            .ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" could not build a transaction-private cold delta",
                    table.name
                )))
            })?;
            cold_chunks.insert(table.name.clone(), next);
            return Ok(());
        }
        let prefix = relational_key_prefix(&table.name);
        match &delta.mutation {
            PreparedMutation::Insert { inserted_rows, .. } => {
                let row_ids = inserted_rows
                    .iter()
                    .map(|(key, _)| {
                        crate::engine_residency::parse_relational_row_id(key, &prefix).ok_or_else(
                            || {
                                ExecuteError::Engine(EngineError::ApplyFailed(
                                    "transaction INSERT produced a non-canonical entity identity"
                                        .to_string(),
                                ))
                            },
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let rows = inserted_rows
                    .iter()
                    .map(|(_, row)| row.clone())
                    .collect::<Vec<_>>();
                self.append_transaction_delta_shard(table, &rows, &row_ids, shards, gpu_reservation)
            }
            PreparedMutation::Update {
                installs,
                updated_old_rows,
                ..
            } => {
                let row_ids = installs
                    .iter()
                    .map(|(_, key, _)| {
                        crate::engine_residency::parse_relational_row_id(key, &prefix).ok_or_else(
                            || {
                                ExecuteError::Engine(EngineError::ApplyFailed(
                                    "transaction UPDATE lost stable entity identity".to_string(),
                                ))
                            },
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if row_ids.len() != updated_old_rows.len() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction UPDATE old/new identity vectors are not parallel".to_string(),
                    )));
                }
                let targets = row_ids
                    .iter()
                    .copied()
                    .zip(updated_old_rows.iter().map(Vec::as_slice))
                    .collect::<Vec<_>>();
                self.tombstone_transaction_rows(
                    table,
                    &targets,
                    boundary,
                    shards,
                    gpu_reservation,
                )?;
                let rows = installs
                    .iter()
                    .map(|(_, _, row)| row.clone())
                    .collect::<Vec<_>>();
                self.append_transaction_delta_shard(table, &rows, &row_ids, shards, gpu_reservation)
            }
            PreparedMutation::Delete { deleted_rows, .. } => {
                if deleted_rows.len() != delta.write_set.rows.len() {
                    return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                        "transaction DELETE row images lost their identity ordering".to_string(),
                    )));
                }
                let targets = delta
                    .write_set
                    .rows
                    .iter()
                    .zip(deleted_rows)
                    .map(|(key, row)| {
                        crate::engine_residency::parse_relational_row_id(&key.row_key, &prefix)
                            .map(|row_id| (row_id, row.as_slice()))
                            .ok_or_else(|| {
                                ExecuteError::Engine(EngineError::ApplyFailed(
                                    "transaction DELETE lost stable entity identity".to_string(),
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                self.tombstone_transaction_rows(table, &targets, boundary, shards, gpu_reservation)
            }
        }
    }

    fn tombstone_transaction_rows(
        &self,
        table: &RelationalTable,
        targets: &[TargetRow<'_>],
        boundary: Index,
        shards: &mut BTreeMap<String, Vec<RelationalResidentShard>>,
        gpu_reservation: &mut TransactionGpuReservation<'_>,
    ) -> Result<(), ExecuteError> {
        let mut slots_by_shard: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for &(row_id, old_row) in targets {
            let filters = old_row
                .iter()
                .enumerate()
                .map(|(idx, value)| (idx, SelectFilterOp::Eq, value.clone()))
                .collect::<Vec<_>>();
            let (key_id, needle) = self.dml_device_probe_key(table, &filters).ok_or_else(|| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "relation \"{}\" has no device key usable by the transaction delta",
                    table.name
                )))
            })?;
            let hits = self
                .locate_resident_pk_via_shard_index_detailed(table, key_id, needle)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "relation \"{}\" transaction target locate declined",
                        table.name
                    )))
                })?;
            let mut visible = Vec::new();
            let mut observed = Vec::new();
            for hit in hits {
                let Some(identity) = self.hit_entity_id(&hit) else {
                    observed.push((hit.shard_id, hit.slot, None, None));
                    continue;
                };
                let materialized = self.materialize_resident_row_via_hit(table, &hit, boundary);
                observed.push((hit.shard_id, hit.slot, Some(identity), materialized.clone()));
                if identity != row_id {
                    continue;
                }
                if materialized.is_some_and(|row| row.as_deref() == Some(old_row)) {
                    visible.push((hit.shard_id, hit.slot));
                }
            }
            if visible.len() != 1 {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "transaction target entity {row_id} resolved {} visible versions; expected {old_row:?}, observed {observed:?}",
                    visible.len(),
                ))));
            }
            let (shard_id, slot) = visible[0];
            slots_by_shard.entry(shard_id).or_default().push(slot);
        }

        let table_shards = shards.get_mut(&table.name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation \"{}\" lost its transaction shard generation",
                table.name
            )))
        })?;
        for (shard_id, mut slots) in slots_by_shard {
            slots.sort_unstable();
            slots.dedup();
            let shard = table_shards
                .iter_mut()
                .find(|shard| shard.shard_id == shard_id)
                .ok_or_else(|| {
                    ExecuteError::Engine(EngineError::ApplyFailed(format!(
                        "transaction target shard {shard_id} disappeared"
                    )))
                })?;
            let private = self.clone_transaction_deleted_region(shard, gpu_reservation)?;
            let stamps = vec![boundary; slots.len()];
            private.scatter_u64_slots(&slots, &stamps).map_err(|err| {
                ExecuteError::Engine(EngineError::ApplyFailed(format!(
                    "transaction tombstone scatter failed: {err}"
                )))
            })?;
            shard.deleted_by_region = Some(private);
        }
        Ok(())
    }

    pub(crate) fn hit_entity_id(
        &self,
        hit: &crate::engine_retained_read::ShardPkHit,
    ) -> Option<u64> {
        let words = hit
            .row_id
            .as_ref()?
            .read_resident_i32_column(u64::from(hit.slot) * 8, 2)
            .ok()?;
        let lo = *words.first()? as u32 as u64;
        let hi = *words.get(1)? as u32 as u64;
        Some(lo | (hi << 32))
    }
}

/// Row mutation decoding, FK closure, and typed value semantics are independent of the table's
/// index list. An ordered transaction may therefore stage DML on both sides of its own index
/// lifecycle commands. Those commands carry their own exact table/index before/after identities;
/// accepting only an index-list difference here avoids making that typed owner compete with a
/// second, stale full-table equality check.
fn same_transaction_row_catalog_dependency(
    expected: &RelationalTable,
    observed: &RelationalTable,
    sequence_input_oids: &BTreeMap<String, u32>,
    transaction_catalog: &CatalogSnapshot,
    catalog_commands: &[StagedCatalogCommand],
) -> bool {
    if expected == observed {
        return true;
    }
    let mut normalized = expected.clone();
    normalized.indexes = observed.indexes.clone();
    for column in &mut normalized.columns {
        let Some(ColumnDefault::SequenceNextVal { sequence, .. }) = &mut column.default else {
            continue;
        };
        let expected_oid = sequence_input_oids.get(sequence).copied().or_else(|| {
            catalog_commands
                .iter()
                .filter_map(|staged| staged.sequence_identity.as_ref())
                .flat_map(|identity| &identity.targets)
                .find_map(|target| {
                    (target.before_name == *sequence)
                        .then(|| target.target_before.as_ref().map(|identity| identity.oid))
                        .flatten()
                })
        });
        let Some(expected_oid) = expected_oid else {
            continue;
        };
        let Some(current_name) = transaction_catalog
            .relational_sequences
            .iter()
            .find_map(|(name, candidate)| (candidate.oid == expected_oid).then_some(name))
        else {
            return false;
        };
        *sequence = current_name.clone();
    }
    &normalized == observed
}
